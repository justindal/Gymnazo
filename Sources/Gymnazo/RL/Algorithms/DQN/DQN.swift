import MLX
import MLXNN
import MLXOptimizers

public actor DQN {
    public nonisolated let config: DQNConfig

    nonisolated(unsafe) private var env: (any Env)?
    private let policy: DQNPolicy
    private let targetPolicy: DQNPolicy
    private var optimizer: Adam
    var buffer: ReplayBuffer?
    let learningRate: any LearningRateSchedule
    let randomSeed: UInt64?

    var explorationRate: Double
    var timesteps: Int = 0
    var totalTimesteps: Int = 0
    var gradientSteps: Int = 0
    var progressRemaining: Double = 1.0

    private var shouldContinue: Bool = true
    private var key: MLXArray
    private var compiledStep: (([MLXArray]) -> [MLXArray])?

    public init(
        env: any Env,
        learningRate: any LearningRateSchedule = ConstantLearningRate(1e-4),
        policyConfig: DQNPolicyConfig = DQNPolicyConfig(),
        config: DQNConfig = DQNConfig(),
        optimizerConfig: DQNOptimizerConfig = DQNOptimizerConfig(),
        seed: UInt64? = nil
    ) throws {
        guard let discrete = env.actionSpace as? Discrete else {
            throw GymnazoError.invalidActionType(
                expected: "Discrete",
                actual: String(describing: type(of: env.actionSpace))
            )
        }

        let networks = DQNNetworks(
            observationSpace: env.observationSpace,
            nActions: discrete.n,
            config: policyConfig
        )

        self.env = env
        self.config = config
        self.policy = networks.qNet
        self.targetPolicy = networks.qNetTarget
        self.learningRate = learningRate
        self.explorationRate = config.explorationInitialEps
        self.randomSeed = seed
        self.key = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))

        let lr = Float(learningRate.value(at: 1.0))
        self.optimizer = optimizerConfig.optimizer.make(learningRate: lr)
    }

    public init(
        observationSpace: any Space,
        actionSpace: Discrete,
        env: (any Env)? = nil,
        learningRate: any LearningRateSchedule = ConstantLearningRate(1e-4),
        policyConfig: DQNPolicyConfig = DQNPolicyConfig(),
        config: DQNConfig = DQNConfig(),
        optimizerConfig: DQNOptimizerConfig = DQNOptimizerConfig(),
        seed: UInt64? = nil
    ) {
        let networks = DQNNetworks(
            observationSpace: observationSpace,
            nActions: actionSpace.n,
            config: policyConfig
        )

        self.env = env
        self.config = config
        self.policy = networks.qNet
        self.targetPolicy = networks.qNetTarget
        self.learningRate = learningRate
        self.explorationRate = config.explorationInitialEps
        self.randomSeed = seed
        self.key = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))

        let lr = Float(learningRate.value(at: 1.0))
        self.optimizer = optimizerConfig.optimizer.make(learningRate: lr)
    }

    init(
        policy: DQNPolicy,
        targetPolicy: DQNPolicy,
        optimizer: Adam,
        config: DQNConfig,
        learningRate: any LearningRateSchedule,
        seed: UInt64?,
        timesteps: Int,
        totalTimesteps: Int,
        progressRemaining: Double,
        explorationRate: Double,
        gradientSteps: Int,
        buffer: ReplayBuffer?
    ) {
        self.env = nil
        self.policy = policy
        self.targetPolicy = targetPolicy
        self.optimizer = optimizer
        self.config = config
        self.learningRate = learningRate
        self.randomSeed = seed
        self.timesteps = timesteps
        self.totalTimesteps = totalTimesteps
        self.progressRemaining = progressRemaining
        self.explorationRate = explorationRate
        self.gradientSteps = gradientSteps
        self.buffer = buffer
        self.key = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
    }

    public func learn(
        totalTimesteps: Int,
        callbacks: LearnCallbacks?,
        resetProgress: Bool = true
    ) async throws {
        self.totalTimesteps = totalTimesteps
        if resetProgress { self.timesteps = 0 }
        shouldContinue = true

        setupBuffer()

        guard var environment = env else {
            throw GymnazoError.invalidState("DQN.learn requires an environment.")
        }

        var lastObs = try environment.reset().obs
        var collectedSteps = 0
        var collectedEpisodes = 0
        var stepsSinceTrain = 0
        var episodeReward: Double = 0
        var episodeLength: Int = 0

        while timesteps < self.totalTimesteps {
            guard shouldContinue else { break }

            updateExplorationRate()

            let action: MLXArray
            let (exploreKey, nextKey) = MLX.split(key: key, stream: .cpu)
            key = nextKey

            if shouldExplore(key: exploreKey) {
                let (sampleKey, nextKey2) = MLX.split(key: key, stream: .cpu)
                key = nextKey2
                action = sampleRandom(key: sampleKey)
            } else {
                action = selectAction(obs: lastObs)
            }

            let step = try environment.step(action)
            let reward = Float(step.reward)
            let terminated = step.terminated
            let truncated = step.truncated

            episodeReward += Double(reward)
            episodeLength += 1

            let bufferNextObs = step.info["final_observation"]?.cast(MLXArray.self) ?? step.obs

            buffer?.add(
                obs: lastObs,
                action: action,
                reward: MLXArray(reward),
                nextObs: bufferNextObs,
                terminated: terminated,
                truncated: truncated
            )

            eval(action, bufferNextObs)

            if terminated || truncated {
                await callbacks?.onEpisodeEnd?(episodeReward, episodeLength)
                episodeReward = 0
                episodeLength = 0
                lastObs = try environment.reset().obs
                collectedEpisodes += 1
            } else {
                lastObs = step.obs
            }

            timesteps += 1
            collectedSteps += 1
            if timesteps >= config.learningStarts { stepsSinceTrain += 1 }
            progressRemaining = 1.0 - Double(timesteps) / Double(self.totalTimesteps)

            if let onStep = callbacks?.onStep {
                let cont = await onStep(timesteps, self.totalTimesteps, explorationRate)
                if !cont { break }
            }

            if let onSnapshot = callbacks?.onSnapshot {
                if let output = try? environment.render() {
                    if case .other(let snapshot) = output {
                        await onSnapshot(snapshot)
                    }
                }
            }

            let shouldTrain: Bool
            switch config.trainFrequency.unit {
            case .step:
                shouldTrain = collectedSteps % config.trainFrequency.frequency == 0
            case .episode:
                shouldTrain = (terminated || truncated)
                    && collectedEpisodes % config.trainFrequency.frequency == 0
            }

            if shouldTrain && timesteps >= config.learningStarts {
                let gradSteps: Int
                switch config.gradientSteps {
                case .fixed(let steps): gradSteps = steps
                case .asCollectedSteps: gradSteps = stepsSinceTrain
                }
                await train(gradientSteps: gradSteps, batchSize: config.batchSize, callbacks: callbacks)
                stepsSinceTrain = 0
            }
        }

        self.env = environment
    }

    public func evaluate(
        episodes: Int,
        deterministic: Bool = true,
        callbacks: EvaluateCallbacks? = nil
    ) async throws {
        guard var environment = env else {
            throw GymnazoError.invalidState("DQN.evaluate requires an environment.")
        }

        for _ in 0..<episodes {
            var obs = try environment.reset().obs
            var done = false
            var episodeReward: Double = 0
            var episodeLength: Int = 0

            while !done {
                let action = predict(observation: obs, deterministic: deterministic)
                let step = try environment.step(action)
                episodeReward += step.reward
                episodeLength += 1
                done = step.terminated || step.truncated
                obs = step.obs

                if let onSnapshot = callbacks?.onSnapshot {
                    if let output = try? environment.render() {
                        if case .other(let snapshot) = output {
                            await onSnapshot(snapshot)
                        }
                    }
                }

                if let onStep = callbacks?.onStep {
                    if !(await onStep()) {
                        self.env = environment
                        return
                    }
                }
            }

            await callbacks?.onEpisodeEnd?(episodeReward, episodeLength)
        }

        self.env = environment
    }

    func predict(observation: MLXArray, deterministic: Bool = true) -> MLXArray {
        if !deterministic {
            let (exploreKey, nextKey) = MLX.split(key: key, stream: .cpu)
            key = nextKey

            if shouldExplore(key: exploreKey) {
                let (sampleKey, nextKey2) = MLX.split(key: key, stream: .cpu)
                key = nextKey2
                let action = sampleRandom(key: sampleKey)
                eval(action)
                return action
            }
        }

        let action = selectAction(obs: observation)
        eval(action)
        return action
    }

    public func stop() {
        shouldContinue = false
    }

    public var numTimesteps: Int { timesteps }
    public var epsilon: Double { explorationRate }
    var qNet: DQNPolicy { policy }
    var qNetTarget: DQNPolicy { targetPolicy }

    nonisolated public func setEnv(_ env: any Env) {
        self.env = env
    }

    nonisolated public func takeEnv() -> (any Env)? {
        let e = env
        env = nil
        return e
    }

    func restore(
        timesteps: Int,
        totalTimesteps: Int,
        progressRemaining: Double,
        explorationRate: Double,
        gradientSteps: Int
    ) {
        self.timesteps = timesteps
        self.totalTimesteps = totalTimesteps
        self.progressRemaining = progressRemaining
        self.explorationRate = explorationRate
        self.gradientSteps = gradientSteps
    }

    private func setupBuffer() {
        guard buffer == nil else { return }
        let bufferConfig = ReplayBuffer.Configuration(
            bufferSize: config.bufferSize,
            optimizeMemoryUsage: config.optimizeMemoryUsage,
            handleTimeoutTermination: config.handleTimeoutTermination,
            seed: randomSeed
        )
        buffer = ReplayBuffer(
            observationSpace: policy.observationSpace,
            actionSpace: policy.actionSpace,
            config: bufferConfig,
            numEnvs: 1
        )
    }

    private func updateExplorationRate() {
        let fraction = min(
            1.0,
            Double(timesteps) / (Double(totalTimesteps) * config.explorationFraction)
        )
        explorationRate = config.explorationInitialEps
            + fraction * (config.explorationFinalEps - config.explorationInitialEps)
    }

    private func shouldExplore(key: MLXArray) -> Bool {
        let random = MLX.uniform(0.0..<1.0, key: key, stream: .cpu)
        eval(random)
        return Double(random.item(Float.self)) < explorationRate
    }

    private func sampleRandom(key: MLXArray) -> MLXArray {
        guard let discrete = policy.actionSpace as? Discrete else {
            preconditionFailure("DQN requires a Discrete action space")
        }
        return discrete.sample(key: key, mask: nil, probability: nil)
    }

    private func selectAction(obs: MLXArray) -> MLXArray {
        policy.setTrainingMode(false)
        let qValues = policy.forward(obs: obs)
        var action = MLX.argMax(qValues, axis: -1, stream: .cpu).asType(.int32, stream: .cpu)
        if action.ndim == 0 { action = action.reshaped([1]) }
        eval(action)
        return action
    }

    private func train(gradientSteps: Int, batchSize: Int, callbacks: LearnCallbacks?) async {
        guard var buf = buffer, buf.count >= batchSize else { return }

        let lr = Float(learningRate.value(at: progressRemaining))
        optimizer.learningRate = lr

        let step = buildCompiledStep()

        var lossArrays: [MLXArray] = []
        var tdArrays: [MLXArray] = []
        var qArrays: [MLXArray] = []
        lossArrays.reserveCapacity(gradientSteps)
        tdArrays.reserveCapacity(gradientSteps)
        qArrays.reserveCapacity(gradientSteps)

        for _ in 0..<gradientSteps {
            let (sampleKey, nextKey) = MLX.split(key: key, stream: .cpu)
            key = nextKey
            let batch = buf.sample(batchSize, key: sampleKey)

            let values = step([batch.obs, batch.actions, batch.nextObs, batch.rewards, batch.dones])
            eval(values[0], values[1], values[2], policy, optimizer)

            lossArrays.append(values[0])
            tdArrays.append(values[1])
            qArrays.append(values[2])

            self.gradientSteps += 1
            if self.gradientSteps % config.targetUpdateInterval == 0 {
                updateTarget()
            }
        }
        buffer = buf

        let totalLoss = lossArrays.reduce(MLXArray(0.0), +)
        let totalTD = tdArrays.reduce(MLXArray(0.0), +)
        let totalQ = qArrays.reduce(MLXArray(0.0), +)
        let steps = Float(gradientSteps)
        eval(totalLoss, totalTD, totalQ)

        await callbacks?.onTrain?([
            "loss": Double((totalLoss / steps).item(Float.self)),
            "tdError": Double((totalTD / steps).item(Float.self)),
            "meanQValue": Double((totalQ / steps).item(Float.self)),
            "learningRate": Double(lr),
        ])
    }

    private func buildCompiledStep() -> ([MLXArray]) -> [MLXArray] {
        if let step = compiledStep { return step }

        let gamma = Float(config.gamma)
        let maxGradNorm = config.maxGradNorm.map { Float($0) }
        let p = policy
        let t = targetPolicy
        let opt = optimizer

        let qNetVG = valueAndGrad(model: p) {
            (model: DQNPolicy, args: [MLXArray]) -> [MLXArray] in
            let (loss, meanAbsTD, meanQ) = DQN.qNetLossAndMetrics(
                model, obs: args[0], actions: args[1], targetQ: args[2]
            )
            return [loss, MLX.stopGradient(meanAbsTD), MLX.stopGradient(meanQ)]
        }

        t.setTrainingMode(false)
        p.setTrainingMode(true)

        let stepBody: ([MLXArray]) -> [MLXArray] = { (args: [MLXArray]) -> [MLXArray] in
            let obs = args[0]
            let actions = args[1]
            let nextObs = args[2]
            let rewards = args[3]
            let dones = args[4]

            let targetQValues = DQN.computeTargetQ(
                target: t, nextObs: nextObs, rewards: rewards, dones: dones, gamma: gamma
            )
            let detachedTargetQ = MLX.stopGradient(targetQValues)

            var (values, grads) = qNetVG(p, [obs, actions, detachedTargetQ])

            if let maxGradNorm {
                grads = DQN.clipGradients(grads, maxNorm: maxGradNorm)
            }

            opt.update(model: p, gradients: grads)
            return values
        }

        let step: ([MLXArray]) -> [MLXArray]
        if p.featuresExtractor is FlattenExtractor {
            step = compile(inputs: [p, t, opt], outputs: [p, opt], stepBody)
        } else {
            step = stepBody
        }

        compiledStep = step
        return step
    }

    private static func computeTargetQ(
        target: DQNPolicy,
        nextObs: MLXArray,
        rewards: MLXArray,
        dones: MLXArray,
        gamma: Float
    ) -> MLXArray {
        let nextQValues = target.forward(obs: nextObs)
        let nextQMax = MLX.max(nextQValues, axis: -1, keepDims: true)
        return rewards.expandedDimensions(axis: -1)
            + (1.0 - dones.expandedDimensions(axis: -1)) * gamma * nextQMax
    }

    private static func qNetLossAndMetrics(
        _ qNet: DQNPolicy,
        obs: MLXArray,
        actions: MLXArray,
        targetQ: MLXArray
    ) -> (loss: MLXArray, tdError: MLXArray, meanQ: MLXArray) {
        let qValues = qNet.forward(obs: obs)
        let batchSize = actions.shape[0]
        let batchIndices = MLXArray(Array(0..<Int32(batchSize)))
        let actionIndices = actions.reshaped([-1]).asType(.int32)
        let selectedQ = qValues[batchIndices, actionIndices].expandedDimensions(axis: -1)
        let td = selectedQ - targetQ
        let loss = MLX.mean(td ** 2)
        let meanAbsTD = MLX.mean(MLX.abs(td))
        let meanQ = MLX.mean(qValues)
        return (loss, meanAbsTD, meanQ)
    }

    private static func clipGradients(_ gradients: ModuleParameters, maxNorm: Float) -> ModuleParameters {
        let flattened = Dictionary(uniqueKeysWithValues: gradients.flattened())
        var totalNormSq = MLXArray(0.0)
        for (_, grad) in flattened {
            totalNormSq = totalNormSq + MLX.sum(grad ** 2)
        }
        let totalNorm = MLX.sqrt(totalNormSq)
        let clipCoef = MLX.minimum(maxNorm / (totalNorm + 1e-6), 1.0)
        var clipped: [String: MLXArray] = [:]
        for (key, grad) in flattened {
            clipped[key] = grad * clipCoef
        }
        return ModuleParameters.unflattened(clipped)
    }

    private func updateTarget() {
        let qNetParams = policy.parameters()
        let targetParams = targetPolicy.parameters()
        let updated = polyakUpdate(target: targetParams, source: qNetParams, tau: config.tau)
        _ = try? targetPolicy.update(parameters: updated, verify: .noUnusedKeys)
        targetPolicy.setTrainingMode(false)
        eval(targetPolicy.parameters())
    }
}
