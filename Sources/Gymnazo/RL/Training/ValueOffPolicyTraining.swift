import MLX

struct ValueActionBatch {
    let actions: [PolicyAction]
    let replay: MLXArray
}

protocol ValueOffPolicyTraining: Actor {
    var timesteps: Int { get set }
    var totalTimesteps: Int { get set }
    var progressRemaining: Double { get set }
    var shouldContinue: Bool { get set }
    var buffer: ReplayBuffer? { get set }
    var randomSeed: UInt64? { get }
    var valueConfig: OffPolicyConfig { get }
    var valueExplorationRate: Double { get }
    var valueObservationShape: [Int]? { get }
    func beginSession() throws
    func endSession()
    func setupBuffer(numEnvs: Int) throws
    func prepareValueCollection(numEnvs: Int) throws
    func valueActions(observations: [TensorValue<Float>], frame: Int) throws -> ValueActionBatch
    func resetValueNoise(dones: [Bool])
    func trainValues(gradientSteps: Int, batchSize: Int, callbacks: LearnCallbacks?) async throws
        -> Bool
}

extension ValueOffPolicyTraining {
    func learnValues<Environment: ValueEnvironment>(
        environments: [Environment],
        totalTimesteps: Int,
        callbacks: LearnCallbacks?,
        resetProgress: Bool
    ) async throws
    where Environment.Observation == TensorValue<Float>, Environment.Action: PolicyActionValue {
        try beginSession()
        defer { endSession() }
        var vector = try SyncValueVectorEnvironment(
            environments: environments, autoresetMode: .sameStep)
        defer { vector.close() }
        let config = valueConfig
        guard vector.numEnvs == 1 || (!config.optimizeMemoryUsage && config.replayFrameStack == nil)
        else {
            throw GymnazoError.invalidConfiguration(
                "Vector training requires independent replay observations without frame compression."
            )
        }
        guard totalTimesteps > 0 else {
            throw GymnazoError.invalidConfiguration("Training timesteps must be positive.")
        }
        self.totalTimesteps = totalTimesteps
        if resetProgress { timesteps = 0 }
        progressRemaining = max(0, 1 - Double(timesteps) / Double(totalTimesteps))
        shouldContinue = true
        try setupBuffer(numEnvs: vector.numEnvs)
        try prepareValueCollection(numEnvs: vector.numEnvs)
        if let info = buffer?.allocationInfo, info.wasCapped {
            await callbacks?.onBufferAllocated?(info)
        }
        var observations = try vector.reset(seed: randomSeed).observations
        var episodeRewards = [Double](repeating: 0, count: vector.numEnvs)
        var episodeLengths = [Int](repeating: 0, count: vector.numEnvs)
        var frames = 0
        var episodesSinceTrain = 0
        var stepsSinceTrain = 0
        while timesteps < totalTimesteps && shouldContinue {
            try Task.checkCancellation()
            try validateTrainingObservations(observations, shape: valueObservationShape)
            let actions = try valueActions(observations: observations, frame: frames)
            let typedActions = try actions.actions.map(Environment.Action.policyValue)
            let result = try vector.step(typedActions)
            let nextObservations = result.observations.indices.map {
                result.finalObservations[$0] ?? result.observations[$0]
            }
            try validateTrainingObservations(nextObservations, shape: valueObservationShape)
            try buffer?.addBatch(
                obs: MLXValueCodec.encodeBatch(observations),
                actions: actions.replay.reshaped([vector.numEnvs, -1]),
                rewards: MLXArray(result.rewards.map(Float.init)),
                nextObs: MLXValueCodec.encodeBatch(nextObservations),
                terminations: result.terminations,
                truncations: result.truncations
            )
            let dones = zip(result.terminations, result.truncations).map { $0 || $1 }
            for index in observations.indices {
                episodeRewards[index] += result.rewards[index]
                episodeLengths[index] += 1
                if dones[index] {
                    await callbacks?.onEpisodeEnd?(episodeRewards[index], episodeLengths[index])
                    episodeRewards[index] = 0
                    episodeLengths[index] = 0
                    episodesSinceTrain += 1
                }
            }
            resetValueNoise(dones: dones)
            observations = result.observations
            let previousTimesteps = timesteps
            timesteps += vector.numEnvs
            stepsSinceTrain += max(0, timesteps - max(previousTimesteps, config.learningStarts))
            frames += 1
            progressRemaining = max(0, 1 - Double(timesteps) / Double(totalTimesteps))
            if let onStep = callbacks?.onStep,
                !(await onStep(timesteps, totalTimesteps, valueExplorationRate))
            {
                break
            }
            guard shouldContinue else { break }
            let shouldTrain =
                config.trainFrequency.unit == .step
                ? frames.isMultiple(of: config.trainFrequency.frequency)
                : episodesSinceTrain >= config.trainFrequency.frequency
            if shouldTrain && timesteps >= config.learningStarts {
                let gradientSteps: Int
                switch config.gradientSteps {
                case .fixed(let count): gradientSteps = count
                case .asCollectedSteps: gradientSteps = stepsSinceTrain
                }
                if try await trainValues(
                    gradientSteps: gradientSteps, batchSize: config.batchSize, callbacks: callbacks)
                {
                    stepsSinceTrain = 0
                    episodesSinceTrain = 0
                }
            }
        }
    }
}

func validateTrainingObservations(_ observations: [TensorValue<Float>], shape: [Int]?) throws {
    guard let shape, observations.allSatisfy({ $0.shape == shape }) else {
        throw GymnazoError.invalidState(
            "Training observations must match the policy observation shape.")
    }
}

func continuousValueActions(_ actions: MLXArray, replay: MLXArray, count: Int) -> ValueActionBatch {
    eval(actions, replay)
    let values = actions.reshaped([-1]).asArray(Float.self)
    let dimension = values.count / count
    return ValueActionBatch(
        actions: (0..<count).map { index in
            .continuous(Array(values[(index * dimension)..<((index + 1) * dimension)]))
        },
        replay: replay
    )
}
