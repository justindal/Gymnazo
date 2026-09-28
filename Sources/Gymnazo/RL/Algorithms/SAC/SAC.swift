import MLX
import MLXNN
import MLXOptimizers

/// Soft Actor-Critic (SAC) — an off-policy, entropy-regularised actor-critic algorithm.
///
/// SAC maximizes a trade-off between expected return and entropy, leading to robust
/// exploration and stable training on continuous action spaces.
///
/// ## Usage
/// ```swift
/// let env = try await Gymnazo.make("Pendulum")
/// let model = SAC(env: env)
/// try await model.learn(totalTimesteps: 100_000, callbacks: nil)
/// let action = model(observation: obs)
/// ```
public actor SAC {
    public nonisolated let offPolicyConfig: OffPolicyConfig
    public nonisolated let networksConfig: SACNetworksConfig
    public nonisolated let optimizerConfig: SACOptimizerConfig

    private var env: (any Env)?
    private let networks: SACNetworks
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

    var policy: SACActor { networks.actor }
    var critic: SACCritic { networks.critic }
    var criticTarget: SACCritic { networks.criticTarget }

    var timesteps: Int = 0
    var totalTimesteps: Int = 0
    var gradientSteps: Int = 0
    var progressRemaining: Double = 1.0

    var shouldContinue: Bool = true
    private var isSessionRunning = false
    var key: MLXArray
    private var compiledStep: (([MLXArray]) -> [MLXArray])?

    /// Creates a SAC agent bound to an environment.
    ///
    /// - Parameters:
    ///   - env: The environment to train in.
    ///   - learningRate: Learning-rate schedule. Defaults to 3e-4.
    ///   - networksConfig: Actor and critic network configuration.
    ///   - config: Off-policy hyper-parameters (buffer, batch size, etc.).
    ///   - optimizerConfig: Per-network optimizer configuration.
    ///   - entCoef: Entropy coefficient mode (automatic tuning or fixed value).
    ///   - targetEntropy: Optional target entropy override. Defaults to `-actionDim`.
    ///   - seed: Optional PRNG seed for reproducibility.
    public init(
        env: sending any Env,
        learningRate: any LearningRateSchedule = ConstantLearningRate(3e-4),
        networksConfig: SACNetworksConfig = SACNetworksConfig(),
        config: OffPolicyConfig = OffPolicyConfig(),
        optimizerConfig: SACOptimizerConfig = SACOptimizerConfig(),
        entCoef: EntropyCoef = .auto(),
        targetEntropy: Float? = nil,
        seed: UInt64? = nil
    ) throws {
        let setup = try Self.buildSetup(
            observationSpace: env.observationSpace,
            actionSpace: env.actionSpace,
            networksConfig: networksConfig,
            optimizerConfig: optimizerConfig,
            entCoef: entCoef,
            learningRate: learningRate,
            targetEntropy: targetEntropy,
            initializationSeed: seed
        )

        self.env = env
        self.offPolicyConfig = config
        self.networksConfig = networksConfig
        self.optimizerConfig = optimizerConfig
        self.networks = setup.networks
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

    /// Creates a SAC agent with explicit spaces, optionally attaching an environment.
    ///
    /// - Parameters:
    ///   - observationSpace: The environment observation space.
    ///   - actionSpace: The environment action space.
    ///   - env: Optional environment. Defaults to `nil`.
    ///   - learningRate: Learning-rate schedule. Defaults to 3e-4.
    ///   - networksConfig: Actor and critic network configuration.
    ///   - config: Off-policy hyper-parameters.
    ///   - optimizerConfig: Per-network optimizer configuration.
    ///   - entCoef: Entropy coefficient mode.
    ///   - targetEntropy: Optional target entropy override.
    ///   - seed: Optional PRNG seed for reproducibility.
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
    ) throws {
        let setup = try Self.buildSetup(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            networksConfig: networksConfig,
            optimizerConfig: optimizerConfig,
            entCoef: entCoef,
            learningRate: learningRate,
            targetEntropy: targetEntropy,
            initializationSeed: seed
        )

        self.env = env
        self.offPolicyConfig = config
        self.networksConfig = networksConfig
        self.optimizerConfig = optimizerConfig
        self.networks = setup.networks
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
        networks: sending SACNetworks,
        actorOptimizer: sending Adam,
        criticOptimizer: sending Adam,
        entropyOptimizer: sending Adam?,
        logEntCoefModule: sending LogEntropyCoefModule,
        offPolicyConfig: OffPolicyConfig,
        networksConfig: SACNetworksConfig,
        optimizerConfig: SACOptimizerConfig,
        entCoefConfig: EntropyCoef,
        targetEntropy: Float,
        learningRate: any LearningRateSchedule,
        seed: UInt64?,
        timesteps: Int,
        totalTimesteps: Int,
        progressRemaining: Double,
        gradientSteps: Int,
        buffer: sending ReplayBuffer?,
        env: sending (any Env)? = nil
    ) {
        let shareFeaturesExtractor = networks.critic.shareFeaturesExtractor
        self.env = env
        self.networks = networks
        self.actorOptimizer = actorOptimizer
        self.criticOptimizer = criticOptimizer
        self.entropyOptimizer = entropyOptimizer
        self.logEntCoefModule = logEntCoefModule
        self.offPolicyConfig = offPolicyConfig
        self.networksConfig = networksConfig
        self.optimizerConfig = optimizerConfig
        self.entCoefConfig = entCoefConfig
        self.targetEntropy = targetEntropy
        self.learningRate = learningRate
        self.randomSeed = seed
        self.shareFeaturesExtractor = shareFeaturesExtractor
        self.timesteps = timesteps
        self.totalTimesteps = totalTimesteps
        self.progressRemaining = progressRemaining
        self.gradientSteps = gradientSteps
        self.buffer = buffer
        self.key = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
    }

    /// Runs the SAC training loop for the specified number of environment steps.
    ///
    /// - Parameters:
    ///   - totalTimesteps: Total environment steps to collect.
    ///   - callbacks: Optional callbacks for step, episode, and train events.
    ///   - resetProgress: If `true` (default), resets the timestep counter before training.
    public func learn(
        totalTimesteps: Int,
        callbacks: LearnCallbacks?,
        resetProgress: Bool = true
    ) async throws {
        try beginSession()
        defer { endSession() }
        self.totalTimesteps = totalTimesteps
        if resetProgress { self.timesteps = 0 }
        shouldContinue = true

        try setupBuffer()
        if let info = buffer?.allocationInfo, info.wasCapped {
            await callbacks?.onBufferAllocated?(info)
        }

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
            try Task.checkCancellation()
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
                bufferAction = try policy.scaleAction(envAction)
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
                envAction = try policy.unscaleAction(bufferAction)
            }

            let step = try environment.step(envAction)
            let reward = Float(step.reward)
            let terminated = step.terminated
            let truncated = step.truncated

            episodeReward += Double(reward)
            episodeLength += 1

            let bufferNextObs =
                step.info[EnvInfoKey.finalObservation]?.cast(MLXArray.self) ?? step.obs

            try buffer?.add(
                obs: lastObs,
                action: bufferAction,
                reward: MLXArray(reward),
                nextObs: bufferNextObs,
                terminated: terminated,
                truncated: truncated
            )

            try MLXEval.evaluate(bufferAction, bufferNextObs)

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
                if gradSteps > 0 {
                    let didTrain = try await train(
                        gradientSteps: gradSteps,
                        batchSize: offPolicyConfig.batchSize,
                        callbacks: callbacks
                    )
                    if didTrain { stepsSinceTrain = 0 }
                }
            }
        }

        self.env = environment
    }

    /// Evaluates the current policy for the given number of episodes.
    ///
    /// - Parameters:
    ///   - episodes: Number of episodes to run.
    ///   - deterministic: If `true` (default), uses the mode action.
    ///   - callbacks: Optional callbacks for step and episode events.
    public func evaluate(
        episodes: Int,
        deterministic: Bool = true,
        callbacks: EvaluateCallbacks? = nil
    ) async throws {
        try beginSession()
        defer { endSession() }
        guard var environment = env else {
            throw GymnazoError.invalidState(
                "SAC.evaluate requires an environment."
            )
        }

        for _ in 0..<episodes {
            try Task.checkCancellation()
            var obs = try environment.reset().obs
            var done = false
            var episodeReward: Double = 0
            var episodeLength: Int = 0

            while !done {
                try Task.checkCancellation()
                let action = inferenceAction(observation: obs, deterministic: deterministic)
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

    /// Returns an action for the given observation.
    ///
    /// - Parameters:
    ///   - observation: The current environment observation.
    ///   - deterministic: If `true` (default), returns the mode action.
    /// - Returns: The unscaled action clipped to the environment action space.
    public func act(
        observation: [Float],
        deterministic: Bool = true
    ) -> PolicyAction {
        let action = inferenceAction(
            observation: MLXArray(observation), deterministic: deterministic)
        return .continuous(action.reshaped([-1]).asArray(Float.self))
    }

    public func act(
        observations: [TensorValue<Float>],
        deterministic: Bool = true
    ) throws -> [PolicyAction] {
        guard !observations.isEmpty else { return [] }
        let batch = try MLXValueCodec.encodeBatch(observations)
        let actions = inferenceAction(observation: batch, deterministic: deterministic)
        let values = actions.reshaped([-1]).asArray(Float.self)
        guard values.count.isMultiple(of: observations.count) else {
            throw GymnazoError.invalidState("SAC returned an invalid action batch.")
        }
        let dimension = values.count / observations.count
        return observations.indices.map { index in
            let start = index * dimension
            return .continuous(Array(values[start..<(start + dimension)]))
        }
    }

    private func inferenceAction(observation: MLXArray, deterministic: Bool) -> MLXArray {
        policy.setTrainingMode(false)
        let action = policy.predict(
            observation: observation,
            deterministic: deterministic
        )
        eval(action)
        return action
    }

    func actionValues(
        observation: [Float],
        deterministic: Bool = true
    ) -> [Float] {
        guard
            case .continuous(let values) = act(
                observation: observation, deterministic: deterministic)
        else {
            preconditionFailure("SAC produces continuous actions.")
        }
        return values
    }

    /// Signals the training loop to stop after the current gradient step completes.
    public func stop() {
        shouldContinue = false
    }

    public var numTimesteps: Int { timesteps }

    func criticTargetParameterValues() -> [Float] {
        let parameters = criticTarget.parameters().flattened().sorted { $0.0 < $1.0 }
        let arrays = parameters.map(\.1)
        eval(arrays)
        return arrays.flatMap { $0.asArray(Float.self) }
    }

    /// Attaches an environment to the agent.
    ///
    /// - Parameter env: The environment to attach.
    public func setEnv(_ environment: sending any Env) throws {
        guard !isSessionRunning else {
            throw GymnazoError.invalidState(
                "Cannot replace an environment during training or evaluation.")
        }
        env = environment
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
        learningRate: any LearningRateSchedule,
        targetEntropy: Float?,
        initializationSeed: UInt64? = nil
    ) throws -> NetworkSetup {
        let networks = try SACNetworks(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            config: networksConfig,
            initializationSeed: initializationSeed
        )
        let lr = Float(learningRate.value(at: 1.0))
        let actorOpt = optimizerConfig.actor.makeScheduled(learningRate: lr, schedule: learningRate)
        let criticOpt = optimizerConfig.critic.makeScheduled(
            learningRate: lr, schedule: learningRate)
        let entOpt: Adam? =
            entCoef.isAuto
            ? optimizerConfig.entropy?.makeScheduled(learningRate: lr, schedule: learningRate) : nil
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

    func setupBuffer(numEnvs: Int = 1) throws {
        guard buffer == nil else { return }
        let bufferConfig = ReplayBuffer.Configuration(
            bufferSize: offPolicyConfig.bufferSize,
            optimizeMemoryUsage: offPolicyConfig.optimizeMemoryUsage,
            handleTimeoutTermination: offPolicyConfig.handleTimeoutTermination,
            frameStack: offPolicyConfig.replayFrameStack,
            seed: randomSeed
        )
        buffer = try ReplayBuffer(
            observationSpace: policy.observationSpace,
            actionSpace: policy.actionSpace,
            config: bufferConfig,
            numEnvs: numEnvs
        )
    }

    func selectAction(obs: MLXArray, key: MLXArray? = nil) -> MLXArray {
        policy.setTrainingMode(false)
        let (action, _) = policy.actionLogProb(obs: obs, key: key)
        eval(action)
        return action
    }

    func actionToMLXArray(_ action: Any) throws -> MLXArray {
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

    @discardableResult
    func train(
        gradientSteps: Int,
        batchSize: Int,
        callbacks: LearnCallbacks?
    ) async throws -> Bool {
        guard gradientSteps > 0 else { return false }
        guard var buf = buffer, buf.count >= batchSize else { return false }
        let targetUpdateInterval = max(1, offPolicyConfig.targetUpdateInterval)

        let lr = Float(learningRate.value(at: progressRemaining))
        actorOptimizer.learningRate = lr
        criticOptimizer.learningRate = lr
        entropyOptimizer?.learningRate = lr

        let step = buildCompiledStep()

        var criticLossArrays: [MLXArray] = []
        var actorLossArrays: [MLXArray] = []
        criticLossArrays.reserveCapacity(gradientSteps)
        actorLossArrays.reserveCapacity(gradientSteps)

        for gradientStep in 0..<gradientSteps {
            try Task.checkCancellation()
            if policy.useSDE && offPolicyConfig.sdeSupported {
                let (noiseKey, k1) = MLX.split(key: key, stream: .cpu)
                key = k1
                policy.resetNoise(key: noiseKey)
            }
            let (actionKey, nextKey) = MLX.split(key: key, stream: .cpu)
            key = nextKey
            let batch = try buf.sample(batchSize)

            let values = step(
                [
                    batch.obs, batch.actions, batch.nextObs,
                    batch.rewards, batch.dones, actionKey,
                ]
            )
            if gradientStep % targetUpdateInterval == 0 {
                try softUpdate()
            }
            var state: [Any] = [
                values, policy, critic, criticTarget,
                actorOptimizer, criticOptimizer, logEntCoefModule,
            ]
            if let entropyOptimizer { state.append(entropyOptimizer) }
            try checkedEval(state)
            criticLossArrays.append(values[0])
            actorLossArrays.append(values[1])
        }
        self.gradientSteps += gradientSteps
        buffer = buf

        guard let onTrain = callbacks?.onTrain else { return true }
        let totalCriticLoss = criticLossArrays.reduce(MLXArray(0.0), +)
        let totalActorLoss = actorLossArrays.reduce(MLXArray(0.0), +)
        try MLXEval.evaluate(totalCriticLoss, totalActorLoss)
        let avgCriticLoss = (totalCriticLoss / Float(max(1, criticLossArrays.count))).scalarValue(
            Float.self)
        let avgActorLoss = (totalActorLoss / Float(max(1, actorLossArrays.count))).scalarValue(
            Float.self)
        let entCoefValue = logEntCoefModule.entCoef.scalarValue(Float.self)

        var metrics: [String: Double] = [
            "learningRate": Double(lr)
        ]
        if avgCriticLoss.isFinite {
            metrics["loss"] = Double(avgCriticLoss)
            metrics["criticLoss"] = Double(avgCriticLoss)
        }
        if avgActorLoss.isFinite {
            metrics["actorLoss"] = Double(avgActorLoss)
        }
        if entCoefValue.isFinite {
            metrics["entCoef"] = Double(entCoefValue)
        }

        if !metrics.isEmpty {
            await onTrain(metrics)
        }
        return true
    }

    private func buildCompiledStep() -> ([MLXArray]) -> [MLXArray] {
        if let step = compiledStep { return step }

        let gamma = Float(offPolicyConfig.gamma)
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
                    targetQ: args[2],
                    shareFeaturesExtractor: shareFeatures
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
                [
                    SACMath.entropyLoss(
                        logEntCoef: model.value,
                        logProb: args[0],
                        targetEntropy: targetEnt
                    )
                ]
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

            let targetQ = SACMath.target(
                rewards: rewards,
                dones: dones,
                minQ: minQ!,
                nextLogProb: MLX.stopGradient(nextLogProb),
                entCoef: entCoef,
                gamma: gamma
            )
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
            let (actorValues, actorGrads) = actorVG(
                p,
                [obs, entCoef, actorKey, criticFeatures]
            )
            actOpt.update(model: p, gradients: actorGrads)

            if let entVG, let entOpt {
                let (_, entGrads) = entVG(entModule, [actorValues[1]])
                entOpt.update(model: entModule, gradients: entGrads)
            }

            return [criticLossArrays[0], actorValues[0]]
        }

        let step: ([MLXArray]) -> [MLXArray]
        let criticSupportsCompilation = c.featuresExtractor.map(supportsCompilation) ?? false
        if policy.useSDE || !supportsCompilation(policy.featuresExtractor)
            || !criticSupportsCompilation
        {
            step = stepBody
        } else {
            var compileInputs: [any Updatable] = [
                p, c, ct, actOpt, critOpt, entModule,
            ]
            var compileOutputs: [any Updatable] = [
                p, c, actOpt, critOpt, entModule,
            ]
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
        targetQ: MLXArray,
        shareFeaturesExtractor: Bool
    ) -> MLXArray {
        let extractedFeatures = critic.extractFeatures(
            obs: obs,
            featuresExtractor: critic.extractor
        )
        let features =
            shareFeaturesExtractor
            ? MLX.stopGradient(extractedFeatures)
            : extractedFeatures
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

    func beginSession() throws {
        guard !isSessionRunning else {
            throw GymnazoError.invalidState("Training or evaluation is already running.")
        }
        isSessionRunning = true
    }

    func endSession() {
        isSessionRunning = false
    }

    private func softUpdate(tau: Double? = nil) throws {
        let t = tau ?? offPolicyConfig.tau
        let criticParams = critic.parameters()
        let targetParams = criticTarget.parameters()
        let updated = polyakUpdate(
            target: targetParams,
            source: criticParams,
            tau: t
        )
        do {
            try criticTarget.update(parameters: updated, verify: .noUnusedKeys)
        } catch {
            throw GymnazoError.operationFailed(
                "Failed to update SAC critic target network: \(error)"
            )
        }
        criticTarget.setTrainingMode(false)
    }
}
