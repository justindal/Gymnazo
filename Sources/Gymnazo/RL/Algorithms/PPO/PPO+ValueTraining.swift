import MLX

extension PPO {
    /// Trains PPO using batched typed environments and per-environment advantages.
    ///
    /// `nSteps` counts vector frames per rollout. Timesteps count individual
    /// transitions; a final full frame may exceed the target by fewer than the
    /// number of environments. Typed environments do not produce snapshot callbacks.
    ///
    /// - Parameters:
    ///   - environments: Environments owned by this training run.
    ///   - totalTimesteps: Target number of individual transitions.
    ///   - callbacks: Optional training, progress, and episode callbacks.
    ///   - resetProgress: Whether to restart progress and update counters.
    /// - Throws: If an environment, action conversion, or configuration is invalid.
    public func learn<Environment: ValueEnvironment>(
        environments: [Environment],
        totalTimesteps: Int,
        callbacks: LearnCallbacks? = nil,
        resetProgress: Bool = true
    ) async throws
    where Environment.Observation == TensorValue<Float>, Environment.Action: PolicyActionValue {
        try beginSession()
        defer { endSession() }
        guard totalTimesteps > 0 else {
            throw GymnazoError.invalidConfiguration("Training timesteps must be positive.")
        }
        var vector = try SyncValueVectorEnvironment(
            environments: environments, autoresetMode: .sameStep)
        defer { vector.close() }
        self.totalTimesteps = totalTimesteps
        if resetProgress {
            timesteps = 0
            updates = 0
        }
        progressRemaining = max(0, 1 - Double(timesteps) / Double(totalTimesteps))
        shouldContinue = true
        var observations = try vector.reset(seed: randomSeed).observations
        var episodeStarts = [Float](repeating: 1, count: vector.numEnvs)
        var episodeRewards = [Double](repeating: 0, count: vector.numEnvs)
        var episodeLengths = [Int](repeating: 0, count: vector.numEnvs)
        var rollout = RolloutBuffer(
            bufferSize: config.nSteps, observationSpace: policy.observationSpace,
            actionSpace: policy.actionSpace, numEnvs: vector.numEnvs
        )
        while timesteps < totalTimesteps && shouldContinue {
            try Task.checkCancellation()
            rollout.reset()
            if config.useSDE {
                policy.resetNoise(nEnvs: vector.numEnvs, key: nextKey(for: &key, stream: .cpu))
            }
            for frame in 0..<config.nSteps {
                guard timesteps < totalTimesteps && shouldContinue else { break }
                try Task.checkCancellation()
                if config.useSDE && config.sdeSampleFreq > 0
                    && frame.isMultiple(of: config.sdeSampleFreq)
                {
                    policy.resetNoise(nEnvs: vector.numEnvs, key: nextKey(for: &key, stream: .cpu))
                }
                policy.setTrainingMode(false)
                try validateTrainingObservations(observations, shape: policy.observationSpace.shape)
                let obs = try MLXValueCodec.encodeBatch(observations)
                let output = policy.forward(
                    obs: obs, deterministic: false, key: nextKey(for: &key, stream: .cpu)
                )
                eval(output.actions, output.values, output.logProb)
                let actions = try policyActions(output.actions, count: vector.numEnvs)
                    .map(Environment.Action.policyValue)
                let result = try vector.step(actions)
                try validateTrainingObservations(
                    result.observations, shape: policy.observationSpace.shape)
                var reward = MLXArray(result.rewards.map(Float.init))
                let bootstrapTruncations = zip(result.terminations, result.truncations).map {
                    PPOMath.shouldBootstrap(terminated: $0, truncated: $1)
                }
                if bootstrapTruncations.contains(true) {
                    let final = result.observations.indices.map {
                        result.finalObservations[$0] ?? result.observations[$0]
                    }
                    try validateTrainingObservations(final, shape: policy.observationSpace.shape)
                    let values = policy.predictValues(obs: try MLXValueCodec.encodeBatch(final))
                        .reshaped([vector.numEnvs])
                    let mask = MLXArray(bootstrapTruncations)
                    reward =
                        reward
                        + MLX.which(mask, Float(config.gamma) * values, MLX.zeros(like: values))
                }
                rollout.append(
                    RolloutStep(
                        observation: obs, action: output.actions, reward: reward,
                        episodeStart: MLXArray(episodeStarts), value: output.values,
                        logProb: output.logProb
                    ))
                for index in observations.indices {
                    episodeRewards[index] += result.rewards[index]
                    episodeLengths[index] += 1
                    let done = result.terminations[index] || result.truncations[index]
                    episodeStarts[index] = done ? 1 : 0
                    if done {
                        await callbacks?.onEpisodeEnd?(episodeRewards[index], episodeLengths[index])
                        episodeRewards[index] = 0
                        episodeLengths[index] = 0
                    }
                }
                observations = result.observations
                timesteps += vector.numEnvs
                progressRemaining = max(0, 1 - Double(timesteps) / Double(totalTimesteps))
                if let onStep = callbacks?.onStep,
                    !(await onStep(timesteps, totalTimesteps, 0))
                {
                    shouldContinue = false
                }
            }
            guard shouldContinue, rollout.count > 0 else { break }
            rollout.computeReturnsAndAdvantages(
                lastValues: policy.predictValues(obs: try MLXValueCodec.encodeBatch(observations)),
                dones: MLXArray(episodeStarts), gamma: config.gamma, gaeLambda: config.gaeLambda
            )
            await train(rolloutBuffer: rollout, callbacks: callbacks)
        }
    }
}
