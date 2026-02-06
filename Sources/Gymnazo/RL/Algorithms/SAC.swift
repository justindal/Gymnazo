//
//  SAC.swift
//  Gymnazo
//

import MLX
import MLXNN
import MLXOptimizers

public final class SAC: OffPolicyAlgorithm, @unchecked Sendable {
    public typealias PolicyType = SACActor

    public var config: OffPolicyConfig
    public var replayBuffer: ReplayBuffer<MLXArray>?

    public let policy: SACActor
    public let critic: SACCritic
    public let criticTarget: SACCritic
    public var env: (any Env)?

    public var learningRate: any LearningRateSchedule
    public var currentProgressRemaining: Double
    public var numTimesteps: Int
    public var totalTimesteps: Int
    private var numGradientSteps: Int = 0

    public var actorOptimizer: Adam
    public var criticOptimizer: Adam
    public var entropyOptimizer: Adam?

    public let entCoefConfig: EntropyCoef
    private var logEntCoefModule: LogEntropyCoefModule
    public var targetEntropy: Float

    public var logEntCoef: MLXArray { logEntCoefModule.value }
    public var entCoefTensor: MLXArray { logEntCoefModule.entCoef }

    private let shareFeaturesExtractor: Bool
    private var randomKey: MLXArray
    private let seed: UInt64?

    public var actor: SACActor { policy }

    /// Convenience init for concrete environments.
    public convenience init(
        env: any Env,
        learningRate: any LearningRateSchedule = ConstantLearningRate(3e-4),
        networksConfig: SACNetworksConfig = SACNetworksConfig(),
        config: OffPolicyConfig = OffPolicyConfig(),
        optimizerConfig: SACOptimizerConfig = SACOptimizerConfig(),
        entCoef: EntropyCoef = .auto(),
        targetEntropy: Float? = nil,
        seed: UInt64? = nil
    ) {
        self.init(
            observationSpace: env.observationSpace,
            actionSpace: env.actionSpace,
            env: env,
            learningRate: learningRate,
            networksConfig: networksConfig,
            config: config,
            optimizerConfig: optimizerConfig,
            entCoef: entCoef,
            targetEntropy: targetEntropy,
            seed: seed
        )
    }

    public init(
        networks: SACNetworks,
        env: (any Env)? = nil,
        learningRate: any LearningRateSchedule = ConstantLearningRate(3e-4),
        config: OffPolicyConfig = OffPolicyConfig(),
        optimizerConfig: SACOptimizerConfig = SACOptimizerConfig(),
        entCoef: EntropyCoef = .auto(),
        targetEntropy: Float? = nil,
        seed: UInt64? = nil
    ) {
        self.randomKey = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
        self.policy = networks.actor
        self.critic = networks.critic
        self.criticTarget = networks.criticTarget
        self.env = env
        self.learningRate = learningRate
        self.config = config
        self.currentProgressRemaining = 1.0
        self.numTimesteps = 0
        self.totalTimesteps = 0
        self.shareFeaturesExtractor = networks.critic.shareFeaturesExtractor
        self.seed = seed

        let lr = Float(learningRate.value(at: 1.0))
        self.actorOptimizer = optimizerConfig.actor.make(learningRate: lr)
        self.criticOptimizer = optimizerConfig.critic.make(learningRate: lr)
        self.entCoefConfig = entCoef
        self.entropyOptimizer =
            entCoef.isAuto ? optimizerConfig.entropy?.make(learningRate: lr) : nil

        let actionDim = getActionDim(self.policy.continuousActionSpace)
        self.targetEntropy = targetEntropy ?? Float(-actionDim)
        self.logEntCoefModule = LogEntropyCoefModule(initialValue: entCoef.initialValue)
    }

    public convenience init(
        policy: SACActor,
        env: (any Env)? = nil,
        learningRate: any LearningRateSchedule = ConstantLearningRate(3e-4),
        criticConfig: SACCriticConfig = SACCriticConfig(),
        config: OffPolicyConfig = OffPolicyConfig(),
        optimizerConfig: SACOptimizerConfig = SACOptimizerConfig(),
        entCoef: EntropyCoef = .auto(),
        targetEntropy: Float? = nil,
        seed: UInt64? = nil
    ) {
        let networks = SACNetworks(actor: policy, criticConfig: criticConfig)
        self.init(
            networks: networks,
            env: env,
            learningRate: learningRate,
            config: config,
            optimizerConfig: optimizerConfig,
            entCoef: entCoef,
            targetEntropy: targetEntropy,
            seed: seed
        )
    }

    public init(
        observationSpace: any Space,
        actionSpace: any Space,
        env: (any Env)? = nil,
        learningRate: any LearningRateSchedule = ConstantLearningRate(3e-4),
        networksConfig: SACNetworksConfig = SACNetworksConfig(),
        config: OffPolicyConfig = OffPolicyConfig(),
        optimizerConfig: SACOptimizerConfig = SACOptimizerConfig(),
        entCoef: EntropyCoef = .auto(),
        targetEntropy: Float? = nil,
        seed: UInt64? = nil
    ) {
        let networks = SACNetworks(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            config: networksConfig
        )
        self.randomKey = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
        self.policy = networks.actor
        self.critic = networks.critic
        self.criticTarget = networks.criticTarget
        self.env = env
        self.learningRate = learningRate
        self.config = config
        self.currentProgressRemaining = 1.0
        self.numTimesteps = 0
        self.totalTimesteps = 0
        self.shareFeaturesExtractor = networks.critic.shareFeaturesExtractor
        self.seed = seed

        let lr = Float(learningRate.value(at: 1.0))
        self.actorOptimizer = optimizerConfig.actor.make(learningRate: lr)
        self.criticOptimizer = optimizerConfig.critic.make(learningRate: lr)
        self.entCoefConfig = entCoef
        self.entropyOptimizer =
            entCoef.isAuto ? optimizerConfig.entropy?.make(learningRate: lr) : nil

        let actionDim = getActionDim(actionSpace)
        self.targetEntropy = targetEntropy ?? Float(-actionDim)
        self.logEntCoefModule = LogEntropyCoefModule(initialValue: entCoef.initialValue)
    }

    public func setupReplayBuffer() {
        guard replayBuffer == nil else { return }
        let bufferConfig = ReplayBuffer<MLXArray>.Configuration(
            bufferSize: config.bufferSize,
            optimizeMemoryUsage: config.optimizeMemoryUsage,
            handleTimeoutTermination: config.handleTimeoutTermination,
            seed: seed
        )
        replayBuffer = ReplayBuffer(
            observationSpace: policy.observationSpace,
            actionSpace: policy.actionSpace,
            config: bufferConfig,
            numEnvs: 1
        )
    }

    @discardableResult
    public func learn(totalTimesteps: Int) throws -> Self {
        try learn(totalTimesteps: totalTimesteps, callbacks: nil)
    }

    @discardableResult
    public func learn(totalTimesteps: Int, callbacks: LearnCallbacks?) throws -> Self {
        self.totalTimesteps = totalTimesteps
        self.numTimesteps = 0

        setupReplayBuffer()

        guard var environment = env else {
            throw GymnazoError.invalidState(
                "SAC.learn requires an environment. Set env before calling learn()."
            )
        }

        var lastObs = try environment.reset().obs
        var numCollectedSteps = 0
        var numCollectedEpisodes = 0
        var stepsSinceLastTrain = 0
        var episodeReward: Double = 0
        var episodeLength: Int = 0

        if config.sdeSupported && actor.useSDE {
            let (noiseKey, nextKey) = MLX.split(key: randomKey)
            randomKey = nextKey
            actor.resetNoise(key: noiseKey)
        }

        while numTimesteps < totalTimesteps {
            let isWarmup = numTimesteps < config.learningStarts

            let envAction: MLXArray
            let bufferAction: MLXArray

            if isWarmup && !config.useSDEAtWarmup {
                let (sampleKey, nextKey) = MLX.split(key: randomKey)
                randomKey = nextKey
                let sampledAction = policy.actionSpace.sample(
                    key: sampleKey,
                    mask: nil,
                    probability: nil
                )
                envAction = try actionToMLXArray(sampledAction)
                bufferAction = actor.scaleAction(envAction)
            } else {
                if config.sdeSupported && actor.useSDE && config.sdeSampleFreq > 0 {
                    if numCollectedSteps % config.sdeSampleFreq == 0 {
                        let (noiseKey, nextKey) = MLX.split(key: randomKey)
                        randomKey = nextKey
                        actor.resetNoise(key: noiseKey)
                    }
                }

                let (actionKey, nextKey) = MLX.split(key: randomKey)
                randomKey = nextKey
                bufferAction = selectAction(obs: lastObs, key: actionKey)
                envAction = actor.unscaleAction(bufferAction)
            }

            let stepResult = try environment.step(envAction)
            let reward = Float(stepResult.reward)
            let terminated = stepResult.terminated
            let truncated = stepResult.truncated

            episodeReward += Double(reward)
            episodeLength += 1

            let bufferNextObs = stepResult.info["final_observation"]?.cast(MLXArray.self) ?? stepResult.obs

            storeTransition(
                obs: lastObs,
                action: bufferAction,
                reward: reward,
                nextObs: bufferNextObs,
                terminated: terminated,
                truncated: truncated
            )

            eval(bufferAction, bufferNextObs)

            if terminated || truncated {
                callbacks?.onEpisodeEnd?(episodeReward, episodeLength)
                episodeReward = 0
                episodeLength = 0

                lastObs = try nextObsAfterEpisodeEnd(
                    stepResult: stepResult, environment: &environment)
                numCollectedEpisodes += 1
                if config.sdeSupported && actor.useSDE {
                    let (noiseKey, nextKey) = MLX.split(key: randomKey)
                    randomKey = nextKey
                    actor.resetNoise(key: noiseKey)
                }
            } else {
                lastObs = stepResult.obs
            }

            numTimesteps += 1
            numCollectedSteps += 1
            if !isWarmup {
                stepsSinceLastTrain += 1
            }
            currentProgressRemaining = 1.0 - Double(numTimesteps) / Double(totalTimesteps)

            if let onStep = callbacks?.onStep {
                let shouldContinue = onStep(numTimesteps, totalTimesteps, 0.0)
                if !shouldContinue {
                    break
                }
            }
            if let onSnapshot = callbacks?.onSnapshot {
                if let output = try? environment.render() {
                    if case .other(let snapshot) = output {
                        onSnapshot(snapshot)
                    }
                }
            }

            let shouldTrain: Bool
            switch config.trainFrequency.unit {
            case .step:
                shouldTrain = numCollectedSteps % config.trainFrequency.frequency == 0
            case .episode:
                shouldTrain =
                    (terminated || truncated)
                    && numCollectedEpisodes % config.trainFrequency.frequency == 0
            }

            if shouldTrain && numTimesteps >= config.learningStarts {
                let gradientSteps: Int
                switch config.gradientSteps {
                case .fixed(let steps):
                    gradientSteps = steps
                case .asCollectedSteps:
                    gradientSteps = stepsSinceLastTrain
                }
                train(gradientSteps: gradientSteps, batchSize: config.batchSize)
                stepsSinceLastTrain = 0
            }
        }

        self.env = environment
        return self
    }

    private func selectAction(obs: MLXArray, key: MLXArray? = nil) -> MLXArray {
        actor.setTrainingMode(false)
        let (action, _) = actor.actionLogProb(obs: obs, key: key)
        eval(action)
        return action
    }

    private func storeTransition(
        obs: MLXArray,
        action: MLXArray,
        reward: Float,
        nextObs: MLXArray,
        terminated: Bool,
        truncated: Bool
    ) {
        replayBuffer?.add(
            obs: obs,
            action: action,
            reward: MLXArray(reward),
            nextObs: nextObs,
            terminated: terminated,
            truncated: truncated
        )
    }

    private func nextObsAfterEpisodeEnd(
        stepResult: Step,
        environment: inout any Env
    ) throws -> MLXArray {
        return try environment.reset().obs
    }

    private func actionToMLXArray(_ action: Any) throws -> MLXArray {
        if let arr = action as? MLXArray {
            return arr
        } else if let floats = action as? [Float] {
            return MLXArray(floats)
        } else if let doubles = action as? [Double] {
            return MLXArray(doubles.map { Float($0) })
        } else if let num = action as? Float {
            return MLXArray([num])
        } else if let num = action as? Double {
            return MLXArray([Float(num)])
        }
        throw GymnazoError.invalidActionType(
            expected: String(describing: MLXArray.self),
            actual: String(describing: type(of: action))
        )
    }

    public func train(gradientSteps: Int, batchSize: Int) {
        guard var buffer = replayBuffer, buffer.count >= batchSize else { return }

        let lr = Float(learningRate.value(at: currentProgressRemaining))
        actorOptimizer.learningRate = lr
        criticOptimizer.learningRate = lr
        entropyOptimizer?.learningRate = lr

        for _ in 0..<gradientSteps {
            let (sampleKey, k1) = MLX.split(key: randomKey)
            let (actionKey, nextKey) = MLX.split(key: k1)
            randomKey = nextKey
            let batch = buffer.sample(batchSize, key: sampleKey)
            trainStep(batch: batch, actionKey: actionKey)
        }
        replayBuffer = buffer
    }

    private func trainStep(batch: ReplayBuffer<MLXArray>.Sample, actionKey: MLXArray) {
        let gamma = Float(config.gamma)
        let entCoef = entCoefTensor
        let (targetKey, actorKey) = MLX.split(key: actionKey)

        actor.setTrainingMode(false)
        criticTarget.setTrainingMode(false)

        let (nextActions, nextLogProb) = actor.actionLogProb(obs: batch.nextObs, key: targetKey)

        let targetFeatures = criticTarget.extractFeatures(
            obs: batch.nextObs, featuresExtractor: criticTarget.extractor)
        let targetQInput = MLX.concatenated(
            [targetFeatures, MLX.stopGradient(nextActions)], axis: -1)

        var minQ: MLXArray? = nil
        for qNet in criticTarget.qNetworks {
            let q = qNet(targetQInput)
            minQ = minQ.map { MLX.minimum($0, q) } ?? q
        }

        let nextQ = minQ!
            - entCoef * MLX.stopGradient(nextLogProb).expandedDimensions(axis: -1)
        let targetQ = batch.rewards.expandedDimensions(axis: -1)
            + (1.0 - batch.dones.expandedDimensions(axis: -1)) * gamma * nextQ

        eval(targetQ)
        let detachedTargetQ = MLX.stopGradient(targetQ)

        critic.setTrainingMode(true)

        typealias CriticArgs = (obs: MLXArray, actions: MLXArray, targetQ: MLXArray)
        let criticVG = valueAndGrad(model: critic) {
            (model: SACCritic, args: CriticArgs) -> [MLXArray] in
            [self.criticLoss(model, obs: args.obs, actions: args.actions, targetQ: args.targetQ)]
        }
        let (_, criticGrads) = criticVG(critic, (batch.obs, batch.actions, detachedTargetQ))
        criticOptimizer.update(model: critic, gradients: criticGrads)
        eval(critic.parameters())

        critic.setTrainingMode(false)
        let criticFeatures = MLX.stopGradient(
            critic.extractFeatures(obs: batch.obs, featuresExtractor: critic.extractor))
        eval(criticFeatures)

        actor.setTrainingMode(true)

        let (actorLossKey, entCoefKey) = MLX.split(key: actorKey)
        typealias ActorArgs = (
            obs: MLXArray, entCoef: MLXArray, key: MLXArray, criticFeatures: MLXArray
        )
        let actorVG = valueAndGrad(model: actor) {
            (model: SACActor, args: ActorArgs) -> [MLXArray] in
            [self.actorLoss(
                model,
                obs: args.obs,
                entCoef: args.entCoef,
                key: args.key,
                criticFeatures: args.criticFeatures
            )]
        }
        var (_, actorGrads) = actorVG(
            actor, (batch.obs, entCoef, actorLossKey, criticFeatures))

        if shareFeaturesExtractor {
            actorGrads = zeroExtractorGradients(actorGrads)
        }
        actorOptimizer.update(model: actor, gradients: actorGrads)
        eval(actor.parameters())

        if entCoefConfig.isAuto, let entOpt = entropyOptimizer {
            actor.setTrainingMode(false)
            let (_, newLogProb) = actor.actionLogProb(obs: batch.obs, key: entCoefKey)
            let detachedLogProb = MLX.stopGradient(newLogProb)
            eval(detachedLogProb)

            let entVG = valueAndGrad(model: logEntCoefModule) {
                (model: LogEntropyCoefModule, _: Int) -> [MLXArray] in
                [MLX.mean(-model.value * (detachedLogProb + self.targetEntropy))]
            }
            let (_, entGrads) = entVG(logEntCoefModule, 0)
            entOpt.update(model: logEntCoefModule, gradients: entGrads)
            eval(logEntCoefModule.parameters())
        }

        numGradientSteps += 1
        if numGradientSteps % config.targetUpdateInterval == 0 {
            softUpdateTarget()
        }
    }

    private func criticLoss(
        _ critic: SACCritic, obs: MLXArray, actions: MLXArray, targetQ: MLXArray
    ) -> MLXArray {
        let features = critic.extractFeatures(
            obs: obs, featuresExtractor: critic.extractor)
        let qInput = MLX.concatenated([features, actions], axis: -1)

        var loss = MLXArray(Float(0.0))
        for qNet in critic.qNetworks {
            let q = qNet(qInput)
            loss = loss + 0.5 * MLX.mean((q - targetQ) ** 2)
        }
        return loss
    }

    private func actorLoss(
        _ actor: SACActor,
        obs: MLXArray,
        entCoef: MLXArray,
        key: MLXArray,
        criticFeatures: MLXArray
    ) -> MLXArray {
        let (actions, logProb) = actor.actionLogProb(obs: obs, key: key)
        let qInput = MLX.concatenated([criticFeatures, actions], axis: -1)

        var minQ: MLXArray? = nil
        for qNet in critic.qNetworks {
            let q = qNet(qInput)
            minQ = minQ.map { MLX.minimum($0, q) } ?? q
        }

        return MLX.mean(entCoef * logProb.expandedDimensions(axis: -1) - minQ!)
    }

    public func softUpdateTarget() {
        softUpdateTarget(tau: config.tau)
    }

    public func softUpdateTarget(tau: Double) {
        let criticParams = critic.parameters()
        let targetParams = criticTarget.parameters()

        let updated = polyakUpdate(target: targetParams, source: criticParams, tau: tau)
        _ = try? criticTarget.update(parameters: updated, verify: .noUnusedKeys)
        criticTarget.setTrainingMode(false)
        eval(criticTarget.parameters())
    }

    private func zeroExtractorGradients(_ gradients: ModuleParameters) -> ModuleParameters {
        let flattened = Dictionary(uniqueKeysWithValues: gradients.flattened())
        var zeroed: [String: MLXArray] = [:]
        for (key, value) in flattened {
            if key.hasPrefix("featuresExtractor") {
                zeroed[key] = MLX.zeros(like: value)
            } else {
                zeroed[key] = value
            }
        }
        return ModuleParameters.unflattened(zeroed)
    }
}
