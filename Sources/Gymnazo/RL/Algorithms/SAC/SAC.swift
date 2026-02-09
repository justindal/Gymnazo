import MLX
import MLXNN
import MLXOptimizers

public actor SAC {
    public nonisolated let offPolicyConfig: OffPolicyConfig

    nonisolated(unsafe) private var env: (any Env)?
    let policy: SACActor
    let critic: SACCritic
    let criticTarget: SACCritic
    private var actorOptimizer: Adam
    private var criticOptimizer: Adam
    private var entropyOptimizer: Adam?
    let entCoefConfig: EntropyCoef
    var logEntCoefModule: LogEntropyCoefModule
    var targetEntropy: Float
    var buffer: ReplayBuffer?
    let learningRate: any LearningRateSchedule
    let randomSeed: UInt64?
    private let shareFeaturesExtractor: Bool

    var timesteps: Int = 0
    var totalTimesteps: Int = 0
    var gradientSteps: Int = 0
    var progressRemaining: Double = 1.0

    private var shouldContinue: Bool = true
    private var key: MLXArray
    private var compiledStep: (([MLXArray]) -> [MLXArray])?

    public init(
        env: any Env,
        learningRate: any LearningRateSchedule = ConstantLearningRate(3e-4),
        networksConfig: SACNetworksConfig = SACNetworksConfig(),
        config: OffPolicyConfig = OffPolicyConfig(),
        optimizerConfig: SACOptimizerConfig = SACOptimizerConfig(),
        entCoef: EntropyCoef = .auto(),
        targetEntropy: Float? = nil,
        seed: UInt64? = nil
    ) {
        let setup = Self.buildSetup(
            observationSpace: env.observationSpace,
            actionSpace: env.actionSpace,
            networksConfig: networksConfig,
            optimizerConfig: optimizerConfig,
            entCoef: entCoef,
            lr: Float(learningRate.value(at: 1.0)),
            targetEntropy: targetEntropy
        )

        self.env = env
        self.offPolicyConfig = config
        self.policy = setup.networks.actor
        self.critic = setup.networks.critic
        self.criticTarget = setup.networks.criticTarget
        self.actorOptimizer = setup.actorOptimizer
        self.criticOptimizer = setup.criticOptimizer
        self.entropyOptimizer = setup.entropyOptimizer
        self.entCoefConfig = entCoef
        self.logEntCoefModule = setup.logEntCoefModule
        self.targetEntropy = setup.targetEntropy
        self.learningRate = learningRate
        self.randomSeed = seed
        self.shareFeaturesExtractor =
            setup.networks.critic.shareFeaturesExtractor
        self.key = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
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
        let setup = Self.buildSetup(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            networksConfig: networksConfig,
            optimizerConfig: optimizerConfig,
            entCoef: entCoef,
            lr: Float(learningRate.value(at: 1.0)),
            targetEntropy: targetEntropy
        )

        self.env = env
        self.offPolicyConfig = config
        self.policy = setup.networks.actor
        self.critic = setup.networks.critic
        self.criticTarget = setup.networks.criticTarget
        self.actorOptimizer = setup.actorOptimizer
        self.criticOptimizer = setup.criticOptimizer
        self.entropyOptimizer = setup.entropyOptimizer
        self.entCoefConfig = entCoef
        self.logEntCoefModule = setup.logEntCoefModule
        self.targetEntropy = setup.targetEntropy
        self.learningRate = learningRate
        self.randomSeed = seed
        self.shareFeaturesExtractor =
            setup.networks.critic.shareFeaturesExtractor
        self.key = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
    }

    init(
        policy: SACActor,
        critic: SACCritic,
        criticTarget: SACCritic,
        actorOptimizer: Adam,
        criticOptimizer: Adam,
        entropyOptimizer: Adam?,
        logEntCoefModule: LogEntropyCoefModule,
        offPolicyConfig: OffPolicyConfig,
        entCoefConfig: EntropyCoef,
        targetEntropy: Float,
        learningRate: any LearningRateSchedule,
        seed: UInt64?,
        timesteps: Int,
        totalTimesteps: Int,
        progressRemaining: Double,
        gradientSteps: Int,
        buffer: ReplayBuffer?
    ) {
        self.env = nil
        self.policy = policy
        self.critic = critic
        self.criticTarget = criticTarget
        self.actorOptimizer = actorOptimizer
        self.criticOptimizer = criticOptimizer
        self.entropyOptimizer = entropyOptimizer
        self.logEntCoefModule = logEntCoefModule
        self.offPolicyConfig = offPolicyConfig
        self.entCoefConfig = entCoefConfig
        self.targetEntropy = targetEntropy
        self.learningRate = learningRate
        self.randomSeed = seed
        self.shareFeaturesExtractor = critic.shareFeaturesExtractor
        self.timesteps = timesteps
        self.totalTimesteps = totalTimesteps
        self.progressRemaining = progressRemaining
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
            throw GymnazoError.invalidState(
                "SAC.learn requires an environment."
            )
        }

        var lastObs = try environment.reset().obs
        var collectedSteps = 0
        var collectedEpisodes = 0
        var stepsSinceTrain = 0
        var episodeReward: Double = 0
        var episodeLength: Int = 0

        if offPolicyConfig.sdeSupported && policy.useSDE {
            let (noiseKey, nextKey) = MLX.split(key: key, stream: .cpu)
            key = nextKey
            policy.resetNoise(key: noiseKey)
        }

        while timesteps < self.totalTimesteps {
            guard shouldContinue else { break }

            let isWarmup = timesteps < offPolicyConfig.learningStarts

            let envAction: MLXArray
            let bufferAction: MLXArray

            if isWarmup && !offPolicyConfig.useSDEAtWarmup {
                let (sampleKey, nextKey) = MLX.split(key: key, stream: .cpu)
                key = nextKey
                let sampledAction = policy.actionSpace.sample(
                    key: sampleKey,
                    mask: nil,
                    probability: nil
                )
                envAction = try actionToMLXArray(sampledAction)
                bufferAction = policy.scaleAction(envAction)
            } else {
                if offPolicyConfig.sdeSupported && policy.useSDE
                    && offPolicyConfig.sdeSampleFreq > 0
                {
                    if collectedSteps % offPolicyConfig.sdeSampleFreq == 0 {
                        let (noiseKey, nextKey) = MLX.split(
                            key: key,
                            stream: .cpu
                        )
                        key = nextKey
                        policy.resetNoise(key: noiseKey)
                    }
                }

                let (actionKey, nextKey) = MLX.split(key: key, stream: .cpu)
                key = nextKey
                bufferAction = selectAction(obs: lastObs, key: actionKey)
                envAction = policy.unscaleAction(bufferAction)
            }

            let step = try environment.step(envAction)
            let reward = Float(step.reward)
            let terminated = step.terminated
            let truncated = step.truncated

            episodeReward += Double(reward)
            episodeLength += 1

            let bufferNextObs =
                step.info["final_observation"]?.cast(MLXArray.self) ?? step.obs

            buffer?.add(
                obs: lastObs,
                action: bufferAction,
                reward: MLXArray(reward),
                nextObs: bufferNextObs,
                terminated: terminated,
                truncated: truncated
            )

            eval(bufferAction, bufferNextObs)

            if terminated || truncated {
                await callbacks?.onEpisodeEnd?(episodeReward, episodeLength)
                episodeReward = 0
                episodeLength = 0
                lastObs = try environment.reset().obs
                collectedEpisodes += 1
                if offPolicyConfig.sdeSupported && policy.useSDE {
                    let (noiseKey, nextKey) = MLX.split(key: key, stream: .cpu)
                    key = nextKey
                    policy.resetNoise(key: noiseKey)
                }
            } else {
                lastObs = step.obs
            }

            timesteps += 1
            collectedSteps += 1
            if !isWarmup { stepsSinceTrain += 1 }
            progressRemaining =
                1.0 - Double(timesteps) / Double(self.totalTimesteps)

            if let onStep = callbacks?.onStep {
                let cont = await onStep(timesteps, self.totalTimesteps, 0.0)
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
            switch offPolicyConfig.trainFrequency.unit {
            case .step:
                shouldTrain =
                    collectedSteps % offPolicyConfig.trainFrequency.frequency
                    == 0
            case .episode:
                shouldTrain =
                    (terminated || truncated)
                    && collectedEpisodes
                        % offPolicyConfig.trainFrequency.frequency == 0
            }

            if shouldTrain && timesteps >= offPolicyConfig.learningStarts {
                let gradSteps: Int
                switch offPolicyConfig.gradientSteps {
                case .fixed(let steps): gradSteps = steps
                case .asCollectedSteps: gradSteps = stepsSinceTrain
                }
                await train(
                    gradientSteps: gradSteps,
                    batchSize: offPolicyConfig.batchSize,
                    callbacks: callbacks
                )
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
            throw GymnazoError.invalidState(
                "SAC.evaluate requires an environment."
            )
        }

        for _ in 0..<episodes {
            var obs = try environment.reset().obs
            var done = false
            var episodeReward: Double = 0
            var episodeLength: Int = 0

            while !done {
                let action = predict(
                    observation: obs,
                    deterministic: deterministic
                )
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

    func predict(observation: MLXArray, deterministic: Bool = true) -> MLXArray
    {
        policy.setTrainingMode(false)
        let action = policy.predict(
            observation: observation,
            deterministic: deterministic
        )
        eval(action)
        return action
    }

    public func stop() {
        shouldContinue = false
    }

    public var numTimesteps: Int { timesteps }

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
        gradientSteps: Int
    ) {
        self.timesteps = timesteps
        self.totalTimesteps = totalTimesteps
        self.progressRemaining = progressRemaining
        self.gradientSteps = gradientSteps
    }

    private struct NetworkSetup {
        let networks: SACNetworks
        let actorOptimizer: Adam
        let criticOptimizer: Adam
        let entropyOptimizer: Adam?
        let logEntCoefModule: LogEntropyCoefModule
        let targetEntropy: Float
    }

    private static func buildSetup(
        observationSpace: any Space,
        actionSpace: any Space,
        networksConfig: SACNetworksConfig,
        optimizerConfig: SACOptimizerConfig,
        entCoef: EntropyCoef,
        lr: Float,
        targetEntropy: Float?
    ) -> NetworkSetup {
        let networks = SACNetworks(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            config: networksConfig
        )
        let actorOpt = optimizerConfig.actor.make(learningRate: lr)
        let criticOpt = optimizerConfig.critic.make(learningRate: lr)
        let entOpt: Adam? =
            entCoef.isAuto
            ? optimizerConfig.entropy?.make(learningRate: lr) : nil
        let actionDim = getActionDim(actionSpace)
        let targetEnt = targetEntropy ?? Float(-actionDim)
        let logEntModule = LogEntropyCoefModule(
            initialValue: entCoef.initialValue
        )

        return NetworkSetup(
            networks: networks,
            actorOptimizer: actorOpt,
            criticOptimizer: criticOpt,
            entropyOptimizer: entOpt,
            logEntCoefModule: logEntModule,
            targetEntropy: targetEnt
        )
    }

    private func setupBuffer() {
        guard buffer == nil else { return }
        let bufferConfig = ReplayBuffer.Configuration(
            bufferSize: offPolicyConfig.bufferSize,
            optimizeMemoryUsage: offPolicyConfig.optimizeMemoryUsage,
            handleTimeoutTermination: offPolicyConfig.handleTimeoutTermination,
            seed: randomSeed
        )
        buffer = ReplayBuffer(
            observationSpace: policy.observationSpace,
            actionSpace: policy.actionSpace,
            config: bufferConfig,
            numEnvs: 1
        )
    }

    private func selectAction(obs: MLXArray, key: MLXArray? = nil) -> MLXArray {
        policy.setTrainingMode(false)
        let (action, _) = policy.actionLogProb(obs: obs, key: key)
        eval(action)
        return action
    }

    private func actionToMLXArray(_ action: Any) throws -> MLXArray {
        if let arr = action as? MLXArray { return arr }
        if let floats = action as? [Float] { return MLXArray(floats) }
        if let doubles = action as? [Double] {
            return MLXArray(doubles.map { Float($0) })
        }
        if let num = action as? Float { return MLXArray([num]) }
        if let num = action as? Double { return MLXArray([Float(num)]) }
        throw GymnazoError.invalidActionType(
            expected: String(describing: MLXArray.self),
            actual: String(describing: type(of: action))
        )
    }

    private func train(
        gradientSteps: Int,
        batchSize: Int,
        callbacks: LearnCallbacks?
    ) async {
        guard var buf = buffer, buf.count >= batchSize else { return }

        let lr = Float(learningRate.value(at: progressRemaining))
        actorOptimizer.learningRate = lr
        criticOptimizer.learningRate = lr
        entropyOptimizer?.learningRate = lr

        let includeTargetUpdate = offPolicyConfig.targetUpdateInterval == 1
        let step = buildCompiledStep(includeTargetUpdate: includeTargetUpdate)

        var lossArrays: [MLXArray] = []
        lossArrays.reserveCapacity(gradientSteps)

        for _ in 0..<gradientSteps {
            let (sampleKey, k1) = MLX.split(key: key, stream: .cpu)
            let (actionKey, nextKey) = MLX.split(key: k1, stream: .cpu)
            key = nextKey
            let batch = buf.sample(batchSize, key: sampleKey)

            let values = step(
                [
                    batch.obs, batch.actions, batch.nextObs,
                    batch.rewards, batch.dones, actionKey,
                ]
            )
            eval(
                values[0],
                policy,
                critic,
                criticTarget,
                actorOptimizer,
                criticOptimizer,
                logEntCoefModule
            )
            lossArrays.append(values[0])

            if !includeTargetUpdate {
                self.gradientSteps += 1
                if self.gradientSteps % offPolicyConfig.targetUpdateInterval
                    == 0
                {
                    softUpdate()
                    eval(criticTarget.parameters())
                }
            }
        }
        if includeTargetUpdate {
            self.gradientSteps += gradientSteps
        }
        buffer = buf

        let totalLoss = lossArrays.reduce(MLXArray(0.0), +)
        eval(totalLoss)
        let avgLoss: Float = (totalLoss / Float(gradientSteps)).item()
        let entCoefValue: Float = logEntCoefModule.entCoef.item()

        await callbacks?.onTrain?([
            "loss": Double(avgLoss),
            "entCoef": Double(entCoefValue),
            "learningRate": Double(lr),
        ])
    }

    private func buildCompiledStep(
        includeTargetUpdate: Bool
    ) -> ([MLXArray]) -> [MLXArray] {
        if let step = compiledStep { return step }

        let gamma = Float(offPolicyConfig.gamma)
        let tau = Float(offPolicyConfig.tau)
        let p = policy
        let c = critic
        let ct = criticTarget
        let actOpt = actorOptimizer
        let critOpt = criticOptimizer
        let entModule = logEntCoefModule
        let entOpt = entropyOptimizer
        let targetEnt = targetEntropy
        let shareFeatures = shareFeaturesExtractor
        let autoEntropy = entCoefConfig.isAuto

        let criticVG = valueAndGrad(model: c) {
            (model: SACCritic, args: [MLXArray]) -> [MLXArray] in
            [
                SAC.criticLoss(
                    model,
                    obs: args[0],
                    actions: args[1],
                    targetQ: args[2]
                )
            ]
        }

        let actorVG = valueAndGrad(model: p) {
            (model: SACActor, args: [MLXArray]) -> [MLXArray] in
            SAC.actorLoss(
                model,
                critic: c,
                obs: args[0],
                entCoef: args[1],
                key: args[2],
                criticFeatures: args[3]
            )
        }

        let entVG:
            (
                (LogEntropyCoefModule, [MLXArray]) -> (
                    [MLXArray], ModuleParameters
                )
            )?
        if autoEntropy {
            entVG = valueAndGrad(model: entModule) {
                (model: LogEntropyCoefModule, args: [MLXArray]) -> [MLXArray] in
                [MLX.mean(-model.value * (args[0] + targetEnt))]
            }
        } else {
            entVG = nil
        }

        let stepBody: ([MLXArray]) -> [MLXArray] = {
            (args: [MLXArray]) -> [MLXArray] in
            let obs = args[0]
            let actions = args[1]
            let nextObs = args[2]
            let rewards = args[3]
            let dones = args[4]
            let actionKey = args[5]

            let entCoef = entModule.entCoef
            let (targetKey, actorKey) = MLX.split(key: actionKey)

            p.setTrainingMode(false)
            ct.setTrainingMode(false)

            let (nextActions, nextLogProb) = p.actionLogProb(
                obs: nextObs,
                key: targetKey
            )

            let targetFeatures = ct.extractFeatures(
                obs: nextObs,
                featuresExtractor: ct.extractor
            )
            let targetQInput = MLX.concatenated(
                [targetFeatures, MLX.stopGradient(nextActions)],
                axis: -1
            )

            var minQ: MLXArray? = nil
            for qNet in ct.qNetworks {
                let q = qNet(targetQInput)
                minQ = minQ.map { MLX.minimum($0, q) } ?? q
            }

            let nextQ =
                minQ!
                - entCoef
                * MLX.stopGradient(nextLogProb).expandedDimensions(axis: -1)
            let targetQ =
                rewards.expandedDimensions(axis: -1)
                + (1.0 - dones.expandedDimensions(axis: -1)) * gamma * nextQ
            let detachedTargetQ = MLX.stopGradient(targetQ)

            c.setTrainingMode(true)
            let (criticLossArrays, criticGrads) = criticVG(
                c,
                [obs, actions, detachedTargetQ]
            )
            critOpt.update(model: c, gradients: criticGrads)

            c.setTrainingMode(false)
            let criticFeatures = MLX.stopGradient(
                c.extractFeatures(obs: obs, featuresExtractor: c.extractor)
            )

            p.setTrainingMode(true)
            var (actorValues, actorGrads) = actorVG(
                p,
                [obs, entCoef, actorKey, criticFeatures]
            )

            if shareFeatures {
                actorGrads = SAC.zeroExtractorGradients(actorGrads)
            }
            actOpt.update(model: p, gradients: actorGrads)

            if let entVG, let entOpt {
                let (_, entGrads) = entVG(entModule, [actorValues[1]])
                entOpt.update(model: entModule, gradients: entGrads)
            }

            if includeTargetUpdate {
                let criticParams = c.parameters()
                let targetParams = ct.parameters()
                let updated = polyakUpdate(
                    target: targetParams,
                    source: criticParams,
                    tau: Double(tau)
                )
                _ = try? ct.update(parameters: updated, verify: .noUnusedKeys)
            }

            return criticLossArrays
        }

        let step: ([MLXArray]) -> [MLXArray]
        if policy.useSDE || !(c.featuresExtractor is FlattenExtractor) {
            step = stepBody
        } else {
            var compileInputs: [any Updatable] = [
                p, c, ct, actOpt, critOpt, entModule,
            ]
            var compileOutputs: [any Updatable] = [
                p, c, actOpt, critOpt, entModule,
            ]
            if includeTargetUpdate {
                compileOutputs.insert(ct, at: 2)
            }
            if let entOpt {
                compileInputs.append(entOpt)
                compileOutputs.append(entOpt)
            }
            step = compile(
                inputs: compileInputs,
                outputs: compileOutputs,
                stepBody
            )
        }

        compiledStep = step
        return step
    }

    private static func criticLoss(
        _ critic: SACCritic,
        obs: MLXArray,
        actions: MLXArray,
        targetQ: MLXArray
    ) -> MLXArray {
        let features = critic.extractFeatures(
            obs: obs,
            featuresExtractor: critic.extractor
        )
        let qInput = MLX.concatenated([features, actions], axis: -1)
        var loss = MLXArray(0.0)
        for qNet in critic.qNetworks {
            let q = qNet(qInput)
            loss = loss + 0.5 * MLX.mean((q - targetQ) ** 2)
        }
        return loss
    }

    private static func actorLoss(
        _ actor: SACActor,
        critic: SACCritic,
        obs: MLXArray,
        entCoef: MLXArray,
        key: MLXArray,
        criticFeatures: MLXArray
    ) -> [MLXArray] {
        let (actions, logProb) = actor.actionLogProb(obs: obs, key: key)
        let qInput = MLX.concatenated([criticFeatures, actions], axis: -1)
        var minQ: MLXArray? = nil
        for qNet in critic.qNetworks {
            let q = qNet(qInput)
            minQ = minQ.map { MLX.minimum($0, q) } ?? q
        }
        let loss = MLX.mean(
            entCoef * logProb.expandedDimensions(axis: -1) - minQ!
        )
        return [loss, MLX.stopGradient(logProb)]
    }

    private func softUpdate(tau: Double? = nil) {
        let t = tau ?? offPolicyConfig.tau
        let criticParams = critic.parameters()
        let targetParams = criticTarget.parameters()
        let updated = polyakUpdate(
            target: targetParams,
            source: criticParams,
            tau: t
        )
        _ = try? criticTarget.update(parameters: updated, verify: .noUnusedKeys)
        criticTarget.setTrainingMode(false)
    }

    private static func zeroExtractorGradients(_ gradients: ModuleParameters)
        -> ModuleParameters
    {
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
