import MLX

extension SAC {
    /// Trains with one batched policy call per typed environment frame.
    ///
    /// Timesteps count individual transitions; the final frame may exceed the target
    /// by fewer than the number of environments. Step-based training frequency counts
    /// vector frames, and `asCollectedSteps` counts individual transitions.
    /// Typed environments do not produce snapshot callbacks.
    ///
    /// - Parameters:
    ///   - environments: Environments owned by this training run.
    ///   - totalTimesteps: Target number of individual transitions.
    ///   - callbacks: Optional training, progress, and episode callbacks.
    ///   - resetProgress: Whether to restart the progress counter.
    /// - Throws: If spaces, replay configuration, or environment operations are invalid.
    public func learn<Environment: ValueEnvironment>(
        environments: [Environment],
        totalTimesteps: Int,
        callbacks: LearnCallbacks? = nil,
        resetProgress: Bool = true
    ) async throws
    where Environment.Observation == TensorValue<Float>, Environment.Action: PolicyActionValue {
        try await learnValues(
            environments: environments, totalTimesteps: totalTimesteps,
            callbacks: callbacks, resetProgress: resetProgress
        )
    }
}

extension SAC: ValueOffPolicyTraining {
    func trainValues(gradientSteps: Int, batchSize: Int, callbacks: LearnCallbacks?) async throws
        -> Bool
    {
        try await train(gradientSteps: gradientSteps, batchSize: batchSize, callbacks: callbacks)
    }

    var valueConfig: OffPolicyConfig { offPolicyConfig }
    var valueExplorationRate: Double { 0 }
    var valueObservationShape: [Int]? { policy.observationSpace.shape }
    func prepareValueCollection(numEnvs: Int) throws {
        if policy.useSDE {
            policy.resetNoise(batchSize: numEnvs, key: nextKey(for: &key, stream: .cpu))
        }
    }
    func resetValueNoise(dones: [Bool]) {
        if policy.useSDE && dones.contains(true) {
            policy.resetNoise(batchSize: dones.count, key: nextKey(for: &key, stream: .cpu))
        }
    }
    func valueActions(observations: [TensorValue<Float>], frame: Int) throws -> ValueActionBatch {
        let batch = try MLXValueCodec.encodeBatch(observations)
        let replay: MLXArray
        let actions: MLXArray
        if timesteps < offPolicyConfig.learningStarts && !offPolicyConfig.useSDEAtWarmup {
            let samples = try observations.map { _ in
                try actionToMLXArray(
                    policy.actionSpace.sample(
                        key: nextKey(for: &key, stream: .cpu), mask: nil, probability: nil
                    )
                ).reshaped([-1])
            }
            actions = MLX.stacked(samples)
            replay = try policy.scaleAction(actions)
        } else {
            if policy.useSDE && offPolicyConfig.sdeSampleFreq > 0
                && frame.isMultiple(of: offPolicyConfig.sdeSampleFreq)
            {
                policy.resetNoise(
                    batchSize: observations.count, key: nextKey(for: &key, stream: .cpu))
            }
            replay = selectAction(obs: batch, key: nextKey(for: &key, stream: .cpu))
            actions = try policy.unscaleAction(replay)
        }
        return continuousValueActions(actions, replay: replay, count: observations.count)
    }
}
