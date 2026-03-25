import MLX
import MLXNN
import MLXOptimizers

/// Twin Delayed DDPG (TD3) — an off-policy algorithm for continuous action spaces.
///
/// TD3 improves on DDPG by maintaining two critic networks to mitigate Q-value
/// overestimation, and by delaying actor updates relative to critic updates for
/// more stable training.
///
/// ## Usage
/// ```swift
/// let env = try await Gymnazo.make("Pendulum")
/// let model = TD3(env: env)
/// try await model.learn(totalTimesteps: 100_000, callbacks: nil)
/// let action = model(observation: obs)
/// ```
public actor TD3 {
    public nonisolated let offPolicyConfig: OffPolicyConfig
    public nonisolated let policyConfig: TD3PolicyConfig
    public nonisolated let algorithmConfig: TD3AlgorithmConfig

    nonisolated(unsafe) private var env: (any Env)?

    let policy: TD3Policy
    let learningRate: any LearningRateSchedule
    let randomSeed: UInt64?
    var buffer: ReplayBuffer?

    var timesteps: Int = 0
    var totalTimesteps: Int = 0
    var gradientSteps: Int = 0
    var progressRemaining: Double = 1.0

    private var shouldContinue: Bool = true
    private var key: MLXArray
    private var actionNoise: (any TD3ActionNoise)?
    private var compiledCriticStep: (([MLXArray]) -> [MLXArray])?
    private var compiledActorStep: (([MLXArray]) -> [MLXArray])?

    var actor: TD3Actor { policy.actor }
    var targetActor: TD3Actor { policy.actorTarget }
    var critic: SACCritic { policy.critic }
    var targetCritic: SACCritic { policy.criticTarget }
    var policyDelay: Int { algorithmConfig.policyDelay }
    var targetPolicyNoise: Float { algorithmConfig.targetPolicyNoise }
    var targetNoiseClip: Float { algorithmConfig.targetNoiseClip }
    var actionNoiseConfig: TD3ActionNoiseConfig? { algorithmConfig.actionNoise }

    /// Creates a TD3 agent bound to an environment.
    ///
    /// - Parameters:
    ///   - env: The environment to train in. Must have a continuous (`Box`) action space.
    ///   - learningRate: Learning-rate schedule. Defaults to 1e-3.
    ///   - policyConfig: Actor and critic network configuration.
    ///   - algorithmConfig: TD3-specific hyper-parameters (policy delay, target noise, etc.).
    ///   - config: Off-policy hyper-parameters (buffer, batch size, etc.).
    ///   - seed: Optional PRNG seed for reproducibility.
    public init(
        env: any Env,
        learningRate: any LearningRateSchedule = ConstantLearningRate(1e-3),
        policyConfig: TD3PolicyConfig = TD3PolicyConfig(),
        algorithmConfig: TD3AlgorithmConfig = TD3AlgorithmConfig(),
        config: OffPolicyConfig = OffPolicyConfig(sdeSupported: false),
        seed: UInt64? = nil
    ) {
        self.init(
            observationSpace: env.observationSpace,
            actionSpace: env.actionSpace,
            env: env,
            learningRate: learningRate,
            policyConfig: policyConfig,
            algorithmConfig: algorithmConfig,
            config: config,
            seed: seed
        )
    }

    /// Creates a TD3 agent with explicit spaces, optionally attaching an environment.
    ///
    /// - Parameters:
    ///   - observationSpace: The environment observation space.
    ///   - actionSpace: The environment action space.
    ///   - env: Optional environment. Defaults to `nil`.
    ///   - learningRate: Learning-rate schedule. Defaults to 1e-3.
    ///   - policyConfig: Actor and critic network configuration.
    ///   - algorithmConfig: TD3-specific hyper-parameters.
    ///   - config: Off-policy hyper-parameters.
    ///   - seed: Optional PRNG seed for reproducibility.
    public init(
        observationSpace: any Space,
        actionSpace: any Space,
        env: (any Env)? = nil,
        learningRate: any LearningRateSchedule = ConstantLearningRate(1e-3),
        policyConfig: TD3PolicyConfig = TD3PolicyConfig(),
        algorithmConfig: TD3AlgorithmConfig = TD3AlgorithmConfig(),
        config: OffPolicyConfig = OffPolicyConfig(sdeSupported: false),
        seed: UInt64? = nil
    ) {
        let resolvedOffPolicyConfig = Self.sanitizeOffPolicyConfig(config)
        self.env = env
        self.offPolicyConfig = resolvedOffPolicyConfig
        self.policyConfig = policyConfig
        self.algorithmConfig = algorithmConfig
        self.policy = TD3Policy(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            learningRateSchedule: learningRate,
            config: policyConfig
        )
        self.learningRate = learningRate
        self.randomSeed = seed
        self.key = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
        self.actionNoise = Self.makeActionNoise(
            from: algorithmConfig.actionNoise
        )
    }

    init(
        policy: TD3Policy,
        offPolicyConfig: OffPolicyConfig,
        policyConfig: TD3PolicyConfig,
        algorithmConfig: TD3AlgorithmConfig,
        learningRate: any LearningRateSchedule,
        seed: UInt64?,
        timesteps: Int,
        totalTimesteps: Int,
        progressRemaining: Double,
        gradientSteps: Int,
        buffer: ReplayBuffer?
    ) {
        let resolvedOffPolicyConfig = Self.sanitizeOffPolicyConfig(
            offPolicyConfig
        )
        self.env = nil
        self.policy = policy
        self.offPolicyConfig = resolvedOffPolicyConfig
        self.policyConfig = policyConfig
        self.algorithmConfig = algorithmConfig
        self.learningRate = learningRate
        self.randomSeed = seed
        self.timesteps = timesteps
        self.totalTimesteps = totalTimesteps
        self.progressRemaining = progressRemaining
        self.gradientSteps = gradientSteps
        self.buffer = buffer
        self.key = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
        self.actionNoise = Self.makeActionNoise(
            from: algorithmConfig.actionNoise
        )
    }

    /// Runs the TD3 training loop for the specified number of environment steps.
    ///
    /// - Parameters:
    ///   - totalTimesteps: Total environment steps to collect.
    ///   - callbacks: Optional callbacks for step, episode, train, and snapshot events.
    ///   - resetProgress: If `true` (default), resets the timestep counter before training.
    public func learn(
        totalTimesteps: Int,
        callbacks: LearnCallbacks?,
        resetProgress: Bool = true
    ) async throws {
        self.totalTimesteps = totalTimesteps
        if resetProgress { self.timesteps = 0 }
        shouldContinue = true
        actionNoise?.reset()

        setupBuffer()

        guard var environment = env else {
            throw GymnazoError.invalidState(
                "TD3.learn requires an environment."
            )
        }

        var lastObs = try environment.reset().obs
        var collectedSteps = 0
        var collectedEpisodes = 0
        var stepsSinceTrain = 0
        var episodeReward: Double = 0
        var episodeLength: Int = 0

        while timesteps < self.totalTimesteps {
            guard shouldContinue else { break }

            let isWarmup = timesteps < offPolicyConfig.learningStarts

            let envAction: MLXArray
            let bufferAction: MLXArray

            if isWarmup {
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
                let (actionKey, nextKey) = MLX.split(key: key, stream: .cpu)
                key = nextKey
                bufferAction = selectAction(
                    obs: lastObs,
                    key: actionKey,
                    deterministic: false
                )
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
                actionNoise?.reset()
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
                    let didTrain = await train(
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
        guard var environment = env else {
            throw GymnazoError.invalidState(
                "TD3.evaluate requires an environment."
            )
        }

        for _ in 0..<episodes {
            var obs = try environment.reset().obs
            var done = false
            var episodeReward: Double = 0
            var episodeLength: Int = 0

            while !done {
                let action = callAsFunction(
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

    /// Returns an action for the given observation.
    ///
    /// When `deterministic` is `true` the mode action is returned with no noise.
    /// When `false`, exploration noise from ``TD3AlgorithmConfig/actionNoise`` is added if configured.
    ///
    /// - Parameters:
    ///   - observation: The current environment observation.
    ///   - deterministic: If `true` (default), returns the mode action without exploration noise.
    /// - Returns: The unscaled action clipped to the environment action space.
    public func callAsFunction(
        observation: MLXArray,
        deterministic: Bool = true
    ) -> MLXArray {
        let scaledAction: MLXArray
        if deterministic {
            scaledAction = selectAction(obs: observation, deterministic: true)
        } else {
            let (noiseKey, nextKey) = MLX.split(key: key, stream: .cpu)
            key = nextKey
            scaledAction = selectAction(
                obs: observation,
                key: noiseKey,
                deterministic: false
            )
        }

        let action = policy.unscaleAction(scaledAction)
        eval(action)
        return action
    }

    /// Signals the training loop to stop at the next step boundary.
    public func stop() {
        shouldContinue = false
    }

    public var numTimesteps: Int { timesteps }

    /// Attaches an environment to the agent.
    ///
    /// - Parameter env: The environment to attach.
    nonisolated public func setEnv(_ env: any Env) {
        self.env = env
    }

    /// Detaches and returns the currently attached environment, leaving the slot empty.
    ///
    /// - Returns: The detached environment, or `nil` if none was attached.
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

    private func selectAction(
        obs: MLXArray,
        key: MLXArray? = nil,
        deterministic: Bool = true
    ) -> MLXArray {
        policy.setTrainingMode(false)
        let features = actor.extractFeatures(
            obs: obs,
            featuresExtractor: actor.featuresExtractor
        )
        var action = actor(features, deterministic: true)
        if !deterministic {
            action = addExplorationNoise(to: action, key: key)
        }
        action = MLX.clip(action, min: -1.0, max: 1.0)
        eval(action)
        return action
    }

    private func addExplorationNoise(to action: MLXArray, key: MLXArray?)
        -> MLXArray
    {
        guard let actionNoise else { return action }
        let noise = actionNoise.sample(shape: action.shape, key: key)
        return MLX.clip(action + noise, min: -1.0, max: 1.0)
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

    @discardableResult
    private func train(
        gradientSteps: Int,
        batchSize: Int,
        callbacks: LearnCallbacks?
    ) async -> Bool {
        guard gradientSteps > 0 else { return false }
        guard var buf = buffer, buf.count >= batchSize else { return false }

        policy.updateOptimizers(progressRemaining: progressRemaining)
        policy.setTrainingMode(true)
        let criticStep = buildCriticStep()
        let actorStep = buildActorStep()

        var criticLossArrays: [MLXArray] = []
        var actorLossArrays: [MLXArray] = []
        criticLossArrays.reserveCapacity(gradientSteps)
        actorLossArrays.reserveCapacity(max(1, gradientSteps / policyDelay))

        for _ in 0..<gradientSteps {
            let (sampleKey, k1) = MLX.split(key: key, stream: .cpu)
            let (targetNoiseKey, nextKey) = MLX.split(key: k1, stream: .cpu)
            key = nextKey

            let batch = buf.sample(batchSize, key: sampleKey)
            let criticValues = criticStep(
                [
                    batch.obs,
                    batch.actions,
                    batch.nextObs,
                    batch.rewards,
                    batch.dones,
                    targetNoiseKey,
                ]
            )
            criticLossArrays.append(criticValues[0])
            eval(criticValues[0], critic, policy.criticOptimizer)

            self.gradientSteps += 1
            if self.gradientSteps % policyDelay == 0 {
                let actorValues = actorStep([batch.obs])
                actorLossArrays.append(actorValues[0])
                softUpdateTargets()
                eval(
                    actorValues[0],
                    actor,
                    targetActor,
                    targetCritic,
                    policy.actorOptimizer
                )
            }
        }

        buffer = buf
        policy.setTrainingMode(false)

        let totalCriticLoss = criticLossArrays.reduce(MLXArray(0.0), +)
        eval(totalCriticLoss)
        let avgCriticLoss =
            (totalCriticLoss / Float(max(1, criticLossArrays.count)))
            .scalarValue(Float.self)

        var metrics: [String: Double] = [
            "criticLoss": Double(avgCriticLoss),
            "learningRate": Double(policy.actorOptimizer.learningRate),
        ]

        if !actorLossArrays.isEmpty {
            let totalActorLoss = actorLossArrays.reduce(MLXArray(0.0), +)
            eval(totalActorLoss)
            let avgActorLoss = (totalActorLoss / Float(actorLossArrays.count))
                .scalarValue(Float.self)
            metrics["actorLoss"] = Double(avgActorLoss)
        }

        await callbacks?.onTrain?(metrics)
        return true
    }

    private func buildCriticStep() -> ([MLXArray]) -> [MLXArray] {
        if let step = compiledCriticStep {
            return step
        }

        let gamma = Float(offPolicyConfig.gamma)
        let targetPolicyNoise = self.targetPolicyNoise
        let targetNoiseClip = self.targetNoiseClip
        let c = critic
        let ta = targetActor
        let tc = targetCritic
        let critOpt = policy.criticOptimizer
        let shareFeaturesExtractor = policy.shareFeaturesExtractor

        let criticVG = valueAndGrad(model: c) {
            (model: SACCritic, args: [MLXArray]) -> [MLXArray] in
            [
                TD3.criticLoss(
                    model,
                    obs: args[0],
                    actions: args[1],
                    targetQ: args[2],
                    shareFeaturesExtractor: shareFeaturesExtractor
                )
            ]
        }

        let stepBody: ([MLXArray]) -> [MLXArray] = {
            (args: [MLXArray]) -> [MLXArray] in
            let obs = args[0]
            let actions = args[1]
            let nextObs = args[2]
            let rewards = args[3]
            let dones = args[4]
            let noiseKey = args[5]

            ta.setTrainingMode(false)
            tc.setTrainingMode(false)

            let features = ta.extractFeatures(
                obs: nextObs,
                featuresExtractor: ta.featuresExtractor
            )
            var nextActions = ta(features, deterministic: true)

            if targetPolicyNoise > 0 {
                var noise =
                    MLX.normal(nextActions.shape, key: noiseKey)
                    * targetPolicyNoise
                if targetNoiseClip > 0 {
                    noise = MLX.clip(
                        noise,
                        min: -targetNoiseClip,
                        max: targetNoiseClip
                    )
                }
                nextActions = nextActions + noise
            }
            nextActions = MLX.clip(nextActions, min: -1.0, max: 1.0)

            let targetQValues = tc(
                obs: nextObs,
                actions: MLX.stopGradient(nextActions)
            )
            var minQ = targetQValues[0]
            for q in targetQValues.dropFirst() {
                minQ = MLX.minimum(minQ, q)
            }

            let targetQ =
                rewards.expandedDimensions(axis: -1)
                + (1.0 - dones.expandedDimensions(axis: -1)) * gamma * minQ
            let detachedTargetQ = MLX.stopGradient(targetQ)

            c.setTrainingMode(true)
            let (criticValues, criticGrads) = criticVG(
                c,
                [obs, actions, detachedTargetQ]
            )
            critOpt.update(model: c, gradients: criticGrads)
            return criticValues
        }

        let step: ([MLXArray]) -> [MLXArray]
        if actor.featuresExtractor is FlattenExtractor,
            let criticExtractor = c.featuresExtractor,
            criticExtractor is FlattenExtractor
        {
            step = compile(
                inputs: [ta, tc, c, critOpt],
                outputs: [c, critOpt],
                stepBody
            )
        } else {
            step = stepBody
        }

        compiledCriticStep = step
        return step
    }

    private func buildActorStep() -> ([MLXArray]) -> [MLXArray] {
        if let step = compiledActorStep {
            return step
        }

        let a = actor
        let c = critic
        let actOpt = policy.actorOptimizer

        let actorVG = valueAndGrad(model: a) {
            (model: TD3Actor, args: [MLXArray]) -> [MLXArray] in
            [TD3.actorLoss(model, critic: c, obs: args[0])]
        }

        let stepBody: ([MLXArray]) -> [MLXArray] = {
            (args: [MLXArray]) -> [MLXArray] in
            let obs = args[0]
            a.setTrainingMode(true)
            c.setTrainingMode(false)
            let (actorValues, actorGrads) = actorVG(a, [obs])
            actOpt.update(model: a, gradients: actorGrads)
            return actorValues
        }

        let step: ([MLXArray]) -> [MLXArray]
        if a.featuresExtractor is FlattenExtractor,
            let criticExtractor = c.featuresExtractor,
            criticExtractor is FlattenExtractor
        {
            step = compile(
                inputs: [a, c, actOpt],
                outputs: [a, actOpt],
                stepBody
            )
        } else {
            step = stepBody
        }

        compiledActorStep = step
        return step
    }

    private func softUpdateTargets() {
        let tau = offPolicyConfig.tau

        let actorParams = actor.parameters()
        let actorTargetParams = targetActor.parameters()
        let updatedActor = polyakUpdate(
            target: actorTargetParams,
            source: actorParams,
            tau: tau
        )
        _ = try? targetActor.update(
            parameters: updatedActor,
            verify: .noUnusedKeys
        )

        let criticParams = critic.parameters()
        let criticTargetParams = targetCritic.parameters()
        let updatedCritic = polyakUpdate(
            target: criticTargetParams,
            source: criticParams,
            tau: tau
        )
        _ = try? targetCritic.update(
            parameters: updatedCritic,
            verify: .noUnusedKeys
        )

        targetActor.setTrainingMode(false)
        targetCritic.setTrainingMode(false)
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
            loss = loss + MLX.mean((q - targetQ) ** 2)
        }
        return loss
    }

    private static func actorLoss(
        _ actor: TD3Actor,
        critic: SACCritic,
        obs: MLXArray
    ) -> MLXArray {
        let actorFeatures = actor.extractFeatures(
            obs: obs,
            featuresExtractor: actor.featuresExtractor
        )
        let actions = actor(actorFeatures, deterministic: true)
        let criticFeatures = MLX.stopGradient(
            critic.extractFeatures(
                obs: obs,
                featuresExtractor: critic.extractor
            )
        )
        let qInput = MLX.concatenated([criticFeatures, actions], axis: -1)
        let qValue = critic.qNetworks[0](qInput)
        return -MLX.mean(qValue)
    }

    private static func sanitizeOffPolicyConfig(_ config: OffPolicyConfig)
        -> OffPolicyConfig
    {
        OffPolicyConfig(
            bufferSize: config.bufferSize,
            learningStarts: config.learningStarts,
            batchSize: config.batchSize,
            tau: config.tau,
            gamma: config.gamma,
            trainFrequency: config.trainFrequency,
            gradientSteps: config.gradientSteps,
            targetUpdateInterval: 1,
            optimizeMemoryUsage: config.optimizeMemoryUsage,
            handleTimeoutTermination: config.handleTimeoutTermination,
            useSDEAtWarmup: false,
            sdeSampleFreq: -1,
            sdeSupported: false
        )
    }

    private static func makeActionNoise(from config: TD3ActionNoiseConfig?)
        -> (any TD3ActionNoise)?
    {
        guard let config else { return nil }
        switch config {
        case .normal(let std):
            guard std > 0 else { return nil }
            return TD3NormalActionNoise(std: std)
        case .ornsteinUhlenbeck(let std, let theta, let dt, let initialNoise):
            guard std > 0 else { return nil }
            return TD3OrnsteinUhlenbeckActionNoise(
                std: std,
                theta: theta,
                dt: dt,
                initialNoise: initialNoise
            )
        }
    }
}
