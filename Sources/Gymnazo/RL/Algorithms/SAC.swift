//
//  SAC.swift
//  Gymnazo
//

import MLX
import MLXNN
import MLXOptimizers

public final class SAC<Environment: Env>: OffPolicyAlgorithm
where Environment.Observation == MLXArray, Environment.Action == MLXArray {
    public typealias PolicyType = SACActor
    public typealias EnvType = Environment

    public var config: OffPolicyConfig
    public var replayBuffer: ReplayBuffer<MLXArray>?

    public let policy: SACActor
    public let critic: SACCritic
    public let criticTarget: SACCritic
    public var env: Environment?

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
    ) throws where Environment == AnyEnv<MLXArray, MLXArray> {
        guard let typed = env as? any Env<MLXArray, MLXArray> else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "Env<MLXArray, MLXArray>",
                actual: String(describing: type(of: env))
            )
        }
        let wrapped = AnyEnv(typed)

        self.init(
            observationSpace: wrapped.observationSpace,
            actionSpace: wrapped.actionSpace,
            env: wrapped,
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
        env: Environment? = nil,
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
        env: Environment? = nil,
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
        observationSpace: any Space<MLXArray>,
        actionSpace: any Space<MLXArray>,
        env: Environment? = nil,
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
            handleTimeoutTermination: config.handleTimeoutTermination
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
                bufferAction = selectAction(observation: lastObs, key: actionKey)
                envAction = actor.unscaleAction(bufferAction)
            }

            let stepResult = try environment.step(envAction)
            let reward = Float(stepResult.reward)
            let terminated = stepResult.terminated
            let truncated = stepResult.truncated

            let bufferNextObs = stepResult.info["final_observation"]?.cast(MLXArray.self) ?? stepResult.obs

            storeTransition(
                obs: lastObs,
                action: bufferAction,
                reward: reward,
                nextObs: bufferNextObs,
                terminated: terminated,
                truncated: truncated
            )

            if terminated || truncated {
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

    private func selectAction(observation: MLXArray, key: MLXArray? = nil) -> MLXArray {
        actor.setTrainingMode(false)
        let (action, _) = actor.actionLogProb(obs: observation, key: key)
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
        stepResult: Step<MLXArray>,
        environment: inout Environment
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
        guard let buffer = replayBuffer, buffer.count >= batchSize else { return }

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
    }

    private func trainStep(batch: ReplayBuffer<MLXArray>.Sample, actionKey: MLXArray) {
        actor.setTrainingMode(true)
        critic.setTrainingMode(true)
        criticTarget.setTrainingMode(false)

        let gamma = Float(config.gamma)
        let entCoef = entCoefTensor

        let (targetKey, actorKey) = MLX.split(key: actionKey)
        let (nextActions, nextLogProb) = actor.actionLogProb(obs: batch.nextObs, key: targetKey)
        let targetQValues = MLX.stopGradient(
            computeTargetQ(
                nextObs: batch.nextObs,
                nextActions: MLX.stopGradient(nextActions),
                nextLogProb: MLX.stopGradient(nextLogProb),
                rewards: batch.rewards,
                dones: batch.dones,
                entCoef: entCoef,
                gamma: gamma
            ))

        typealias CriticArgs = (obs: MLXArray, actions: MLXArray, targetQ: MLXArray)
        let criticVG = valueAndGrad(model: critic) {
            (model: SACCritic, args: CriticArgs) -> [MLXArray] in
            [self.criticLoss(model, obs: args.obs, actions: args.actions, targetQ: args.targetQ)]
        }
        let (_, criticGrads) = criticVG(critic, (batch.obs, batch.actions, targetQValues))
        criticOptimizer.update(model: critic, gradients: criticGrads)

        MLX.eval(critic.parameters())

        let (actorLossKey, entCoefKey) = MLX.split(key: actorKey)
        typealias ActorArgs = (obs: MLXArray, entCoef: MLXArray, key: MLXArray)
        let actorVG = valueAndGrad(model: actor) {
            (model: SACActor, args: ActorArgs) -> [MLXArray] in
            [self.actorLoss(model, obs: args.obs, entCoef: args.entCoef, key: args.key)]
        }
        var (_, actorGrads) = actorVG(actor, (batch.obs, entCoef, actorLossKey))

        if shareFeaturesExtractor {
            actorGrads = zeroExtractorGradients(actorGrads)
        }
        actorOptimizer.update(model: actor, gradients: actorGrads)
        MLX.eval(actor.parameters())

        if entCoefConfig.isAuto, let entOpt = entropyOptimizer {
            let (_, newLogProb) = actor.actionLogProb(obs: batch.obs, key: entCoefKey)
            let detachedLogProb = MLX.stopGradient(newLogProb)

            let entVG = valueAndGrad(model: logEntCoefModule) {
                (model: LogEntropyCoefModule, _: Int) -> [MLXArray] in
                [MLX.mean(-model.value * (detachedLogProb + self.targetEntropy))]
            }
            let (_, entGrads) = entVG(logEntCoefModule, 0)
            entOpt.update(model: logEntCoefModule, gradients: entGrads)
            MLX.eval(logEntCoefModule.parameters())
        }

        numGradientSteps += 1
        if numGradientSteps % config.targetUpdateInterval == 0 {
            softUpdateTarget()
        }
    }

    private func computeTargetQ(
        nextObs: MLXArray,
        nextActions: MLXArray,
        nextLogProb: MLXArray,
        rewards: MLXArray,
        dones: MLXArray,
        entCoef: MLXArray,
        gamma: Float
    ) -> MLXArray {
        let features = criticTarget.extractFeatures(
            obs: nextObs,
            featuresExtractor: criticTarget.extractor
        )
        let qInput = MLX.concatenated([features, nextActions], axis: -1)

        var minQ: MLXArray? = nil
        for qNet in criticTarget.qNetworks {
            let q = qNet(qInput)
            minQ = minQ.map { MLX.minimum($0, q) } ?? q
        }

        let nextQ = minQ! - entCoef * nextLogProb.expandedDimensions(axis: -1)
        return rewards.expandedDimensions(axis: -1) + (1.0 - dones.expandedDimensions(axis: -1))
            * gamma * nextQ
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
        key: MLXArray
    ) -> MLXArray {
        let (actions, logProb) = actor.actionLogProb(obs: obs, key: key)
        let features = critic.extractFeatures(
            obs: obs, featuresExtractor: critic.extractor)
        let qInput = MLX.concatenated([features, actions], axis: -1)

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
        MLX.eval(criticTarget.parameters())
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
