import MLX

extension TD3 {
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

extension TD3: ValueOffPolicyTraining {
    func trainValues(gradientSteps: Int, batchSize: Int, callbacks: LearnCallbacks?) async throws
        -> Bool
    {
        try await train(gradientSteps: gradientSteps, batchSize: batchSize, callbacks: callbacks)
    }

    var valueConfig: OffPolicyConfig { offPolicyConfig }
    var valueExplorationRate: Double { 0 }
    var valueObservationShape: [Int]? { policy.observationSpace.shape }
    func prepareValueCollection(numEnvs: Int) throws {
        actionNoise?.reset()
    }
    func resetValueNoise(dones: [Bool]) {
        actionNoise?.reset(indices: dones)
    }
    func valueActions(observations: [TensorValue<Float>], frame: Int) throws -> ValueActionBatch {
        let batch = try MLXValueCodec.encodeBatch(observations)
        let replay: MLXArray
        let actions: MLXArray
        if timesteps < offPolicyConfig.learningStarts {
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
            replay = selectAction(
                obs: batch, key: nextKey(for: &key, stream: .cpu), deterministic: false)
            actions = try policy.unscaleAction(replay)
        }
        return continuousValueActions(actions, replay: replay, count: observations.count)
    }
}
