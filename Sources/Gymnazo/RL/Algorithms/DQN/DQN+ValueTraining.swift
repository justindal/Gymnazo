import MLX

extension DQN {
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

extension DQN: ValueOffPolicyTraining {
    var valueConfig: OffPolicyConfig {
        OffPolicyConfig(
            bufferSize: config.bufferSize, learningStarts: config.learningStarts,
            batchSize: config.batchSize, trainFrequency: config.trainFrequency,
            gradientSteps: config.gradientSteps, optimizeMemoryUsage: config.optimizeMemoryUsage,
            handleTimeoutTermination: config.handleTimeoutTermination,
            replayFrameStack: config.replayFrameStack
        )
    }
    var valueExplorationRate: Double { explorationRate }
    func prepareValueCollection(numEnvs: Int) throws {}
    func resetValueNoise(dones: [Bool]) {}
    func valueActions(observations: [TensorValue<Float>], frame: Int) throws -> ValueActionBatch {
        updateExplorationRate()
        let actions = try act(observations: observations, deterministic: false)
        let values = try actions.map { Float(try Int.policyValue(from: $0)) }
        return ValueActionBatch(actions: actions, replay: MLXArray(values, [values.count, 1]))
    }
}
