import MLX
import MLXNN
import MLXOptimizers

/// Proximal Policy Optimization (PPO) — an on-policy actor-critic algorithm.
///
/// PPO collects fixed-length rollout trajectories and optimizes a clipped surrogate
/// objective to constrain policy updates. It supports discrete, continuous,
/// multi-discrete, and multi-binary action spaces, and optionally State-Dependent
/// Exploration (SDE).
///
/// ## Usage
/// ```swift
/// let env = try await Gymnazo.make("CartPole")
/// let model = try PPO(env: env)
/// try await model.learn(totalTimesteps: 100_000, callbacks: nil)
/// let action = model(observation: obs)
/// ```
public actor PPO {
    public nonisolated let config: PPOConfig
    public nonisolated let policyConfig: PPOPolicyConfig

    private var env: (any Env)?
    let policy: PPOPolicy
    private var optimizer: Adam
    let learningRate: any LearningRateSchedule
    let randomSeed: UInt64?

    var timesteps: Int = 0
    var totalTimesteps: Int = 0
    var updates: Int = 0
    var progressRemaining: Double = 1.0

    var shouldContinue: Bool = true
    private var isSessionRunning = false
    var key: MLXArray
    private var batchGenerator: SeededGenerator
    private var compiledTrainStep: (([MLXArray]) -> [MLXArray])?

    /// Creates a PPO agent bound to an environment.
    ///
    /// - Parameters:
    ///   - env: The environment to train in.
    ///   - learningRate: Learning-rate schedule. Defaults to 3e-4.
    ///   - policyConfig: Network architecture and initialization options.
    ///   - config: PPO hyper-parameters.
    ///   - seed: Optional PRNG seed for reproducibility.
    public init(
        env: sending any Env,
        learningRate: any LearningRateSchedule = ConstantLearningRate(3e-4),
        policyConfig: PPOPolicyConfig = PPOPolicyConfig(),
        config: PPOConfig = PPOConfig(),
        seed: UInt64? = nil
    ) throws {
        try self.init(
            observationSpace: env.observationSpace,
            actionSpace: env.actionSpace,
            env: env,
            learningRate: learningRate,
            policyConfig: policyConfig,
            config: config,
            seed: seed
        )
    }

    /// Creates a PPO agent with explicit spaces, optionally attaching an environment.
    ///
    /// Use this init to restore a saved agent without an environment, then attach one later
    /// via ``setEnv(_:)`` before calling ``learn(totalTimesteps:callbacks:resetProgress:)``.
    ///
    /// - Parameters:
    ///   - observationSpace: The environment observation space.
    ///   - actionSpace: The environment action space.
    ///   - env: Optional environment. Defaults to `nil`.
    ///   - learningRate: Learning-rate schedule. Defaults to 3e-4.
    ///   - policyConfig: Network architecture and initialization options.
    ///   - config: PPO hyper-parameters.
    ///   - seed: Optional PRNG seed for reproducibility.
    public init(
        observationSpace: any Space,
        actionSpace: any Space,
        env: (any Env)? = nil,
        learningRate: any LearningRateSchedule = ConstantLearningRate(3e-4),
        policyConfig: PPOPolicyConfig = PPOPolicyConfig(),
        config: PPOConfig = PPOConfig(),
        seed: UInt64? = nil
    ) throws {
        if config.useSDE && boxSpace(from: actionSpace) == nil {
            throw GymnazoError.invalidConfiguration("PPO useSDE requires a Box action space.")
        }

        self.env = env
        self.config = config
        self.policyConfig = policyConfig
        self.policy = try PPOPolicy(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            config: policyConfig,
            useSDE: config.useSDE,
            initializationKey: seed.map(MLX.key)
        )
        self.learningRate = learningRate
        self.randomSeed = seed
        self.key = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
        self.batchGenerator = SeededGenerator(
            seed: (seed ?? UInt64.random(in: 0..<UInt64.max)) ^ 0xA24B_AED4_963E_E407)

        let optimizerConfig: OptimizerConfig = .adam()
        self.optimizer = optimizerConfig.makeScheduled(
            learningRate: Float(learningRate.value(at: 1.0)), schedule: learningRate
        )
    }

    init(
        policy: sending PPOPolicy,
        optimizer: sending Adam,
        config: PPOConfig,
        policyConfig: PPOPolicyConfig,
        learningRate: any LearningRateSchedule,
        seed: UInt64?,
        timesteps: Int,
        totalTimesteps: Int,
        progressRemaining: Double,
        updates: Int,
        env: sending (any Env)? = nil
    ) {
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.config = config
        self.policyConfig = policyConfig
        self.learningRate = learningRate
        self.randomSeed = seed
        self.timesteps = timesteps
        self.totalTimesteps = totalTimesteps
        self.progressRemaining = progressRemaining
        self.updates = updates
        self.key = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
        self.batchGenerator = SeededGenerator(
            seed: (seed ?? UInt64.random(in: 0..<UInt64.max)) ^ 0xA24B_AED4_963E_E407)
    }

    /// Runs the PPO training loop for the specified number of environment steps.
    ///
    /// - Parameters:
    ///   - totalTimesteps: Total environment steps to collect.
    ///   - callbacks: Optional callbacks for step, episode, train, and snapshot events.
    ///   - resetProgress: If `true` (default), resets the timestep counter, update counter,
    ///     and progress remaining before training.
    public func learn(
        totalTimesteps: Int,
        callbacks: LearnCallbacks?,
        resetProgress: Bool = true
    ) async throws {
        try beginSession()
        defer { endSession() }
        self.totalTimesteps = totalTimesteps
        if resetProgress {
            self.timesteps = 0
            self.updates = 0
            self.progressRemaining = 1.0
        }
        shouldContinue = true

        guard var environment = env else {
            throw GymnazoError.invalidState("PPO.learn requires an environment.")
        }

        var rolloutBuffer = RolloutBuffer(
            bufferSize: config.nSteps,
            observationSpace: policy.observationSpace,
            actionSpace: policy.actionSpace,
            numEnvs: 1
        )

        var lastObs = try environment.reset().obs
        var lastEpisodeStart: Float = 1.0
        var episodeReward: Double = 0.0
        var episodeLength: Int = 0

        while timesteps < self.totalTimesteps {
            try Task.checkCancellation()
            guard shouldContinue else { break }
            rolloutBuffer.reset()

            let rolloutSteps = min(config.nSteps, self.totalTimesteps - timesteps)
            if rolloutSteps <= 0 { break }

            if config.useSDE && config.sdeSampleFreq <= 0 {
                let (noiseKey, nextKey) = MLX.split(key: key, stream: .cpu)
                key = nextKey
                policy.resetNoise(nEnvs: 1, key: noiseKey)
            }

            for stepIndex in 0..<rolloutSteps {
                try Task.checkCancellation()
                guard shouldContinue else { break }

                if config.useSDE && config.sdeSampleFreq > 0
                    && stepIndex % config.sdeSampleFreq == 0
                {
                    let (noiseKey, nextKey) = MLX.split(key: key, stream: .cpu)
                    key = nextKey
                    policy.resetNoise(nEnvs: 1, key: noiseKey)
                }

                policy.setTrainingMode(false)
                let (actionKey, nextKey) = MLX.split(key: key, stream: .cpu)
                key = nextKey
                let output = policy.forward(
                    obs: lastObs,
                    deterministic: false,
                    key: actionKey
                )

                let envAction = actionForEnvironment(output.actions)
                let step = try environment.step(envAction)

                var terminalValue: Float = 0
                if PPOMath.shouldBootstrap(terminated: step.terminated, truncated: step.truncated) {
                    let finalObservation =
                        step.info[EnvInfoKey.finalObservation]?.cast(MLXArray.self) ?? step.obs
                    let bootstrapValue = policy.predictValues(obs: finalObservation)
                    terminalValue = Self.flatten(bootstrapValue).singletonValue(Float.self)
                }
                let adjustedReward = PPOMath.reward(
                    Float(step.reward),
                    terminated: step.terminated,
                    truncated: step.truncated,
                    gamma: config.gamma,
                    terminalValue: terminalValue
                )

                let rolloutStep = RolloutStep(
                    observation: lastObs,
                    action: output.actions,
                    reward: MLXArray(adjustedReward),
                    episodeStart: MLXArray(lastEpisodeStart),
                    value: output.values,
                    logProb: output.logProb
                )
                rolloutBuffer.append(rolloutStep)

                episodeReward += Double(step.reward)
                episodeLength += 1

                let done = step.terminated || step.truncated
                if done {
                    await callbacks?.onEpisodeEnd?(episodeReward, episodeLength)
                    episodeReward = 0.0
                    episodeLength = 0
                    lastObs = try environment.reset().obs
                    lastEpisodeStart = 1.0
                } else {
                    lastObs = step.obs
                    lastEpisodeStart = 0.0
                }

                timesteps += 1
                progressRemaining = 1.0 - Double(timesteps) / Double(self.totalTimesteps)

                if let onStep = callbacks?.onStep {
                    let shouldKeepRunning = await onStep(timesteps, self.totalTimesteps, 0.0)
                    if !shouldKeepRunning {
                        shouldContinue = false
                        break
                    }
                }

                if let onSnapshot = callbacks?.onSnapshot {
                    if let output = try? environment.render() {
                        if case .other(let snapshot) = output {
                            await onSnapshot(snapshot)
                        }
                    }
                }
            }

            guard shouldContinue else { break }
            guard rolloutBuffer.count > 0 else { break }

            let lastValues = policy.predictValues(obs: lastObs)
            rolloutBuffer.computeReturnsAndAdvantages(
                lastValues: lastValues,
                dones: MLXArray(lastEpisodeStart),
                gamma: config.gamma,
                gaeLambda: config.gaeLambda
            )
            await train(rolloutBuffer: rolloutBuffer, callbacks: callbacks)
        }

        self.env = environment
    }

    /// Evaluates the current policy for the given number of episodes.
    ///
    /// - Parameters:
    ///   - episodes: Number of episodes to evaluate.
    ///   - deterministic: If `true` (default), uses the mode action instead of sampling.
    ///   - callbacks: Optional callbacks for step and episode events.
    public func evaluate(
        episodes: Int,
        deterministic: Bool = true,
        callbacks: EvaluateCallbacks? = nil
    ) async throws {
        try beginSession()
        defer { endSession() }
        guard var environment = env else {
            throw GymnazoError.invalidState("PPO.evaluate requires an environment.")
        }

        for _ in 0..<episodes {
            try Task.checkCancellation()
            var obs = try environment.reset().obs
            var done = false
            var episodeReward: Double = 0.0
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
    /// - Returns: The selected action as an `MLXArray` clipped to the action space.
    public func act(
        observation: [Float],
        deterministic: Bool = true
    ) -> PolicyAction {
        let envAction = inferenceAction(
            observation: MLXArray(observation), deterministic: deterministic)
        if policy.actionSpace is Discrete {
            return .discrete(Int(envAction.reshaped([-1]).asArray(Int32.self)[0]))
        }
        if policy.actionSpace is MultiDiscrete {
            return .multiDiscrete(envAction.reshaped([-1]).asArray(Int32.self).map(Int.init))
        }
        if policy.actionSpace is MultiBinary {
            return .multiBinary(envAction.reshaped([-1]).asArray(Float.self).map { $0 >= 0.5 })
        }
        return .continuous(envAction.reshaped([-1]).asArray(Float.self))
    }

    public func act(
        observations: [TensorValue<Float>],
        deterministic: Bool = true
    ) throws -> [PolicyAction] {
        guard !observations.isEmpty else { return [] }
        let batch = try MLXValueCodec.encodeBatch(observations)
        policy.setTrainingMode(false)
        let raw = policy.forward(
            obs: batch,
            deterministic: deterministic,
            key: nil
        ).actions
        return try policyActions(raw, count: observations.count)
    }

    func policyActions(_ raw: MLXArray, count: Int) throws -> [PolicyAction] {

        if let box = boxSpace(from: policy.actionSpace) {
            var actions = raw.asType(.float32)
            if policy.squashOutput {
                actions = (try? policy.unscaleAction(actions)) ?? actions
            }
            actions = MLX.clip(actions, min: box.low, max: box.high)
            eval(actions)
            return try Self.continuousActions(
                actions.reshaped([-1]).asArray(Float.self),
                count: count
            )
        }
        if policy.actionSpace is Discrete {
            let actions = raw.asType(.int32, stream: .cpu).reshaped([-1])
            eval(actions)
            let values = actions.asArray(Int32.self)
            guard values.count == count else {
                throw GymnazoError.invalidState("PPO returned an invalid discrete action batch.")
            }
            return values.map { .discrete(Int($0)) }
        }
        if let multiDiscrete = policy.actionSpace as? MultiDiscrete {
            let dimension = multiDiscrete.nvec.size
            let high = multiDiscrete.nvec.asType(.int32).reshaped([1, dimension]) - 1
            let low = MLX.zeros(high.shape, dtype: .int32)
            let actions = MLX.clip(
                raw.asType(.int32).reshaped([count, dimension]),
                min: low,
                max: high
            )
            eval(actions)
            let values = actions.asArray(Int32.self)
            return (0..<count).map { index in
                let start = index * dimension
                return .multiDiscrete(
                    values[start..<(start + dimension)].map(Int.init)
                )
            }
        }
        if let multiBinary = policy.actionSpace as? MultiBinary,
            let shape = multiBinary.shape
        {
            let dimension = try TensorValue<Bool>.elementCount(for: shape)
            let actions = (raw.asType(.float32) .>= 0.5).reshaped([count, dimension])
            eval(actions)
            let values = actions.asArray(Bool.self)
            return (0..<count).map { index in
                let start = index * dimension
                return .multiBinary(Array(values[start..<(start + dimension)]))
            }
        }
        throw GymnazoError.unsupportedSpaceOperation(
            operation: "batched PPO action decoding",
            spaceType: String(describing: type(of: policy.actionSpace))
        )
    }

    private static func continuousActions(
        _ values: [Float],
        count: Int
    ) throws -> [PolicyAction] {
        guard values.count.isMultiple(of: count) else {
            throw GymnazoError.invalidState("PPO returned an invalid continuous action batch.")
        }
        let dimension = values.count / count
        return (0..<count).map { index in
            let start = index * dimension
            return .continuous(Array(values[start..<(start + dimension)]))
        }
    }

    private func inferenceAction(observation: MLXArray, deterministic: Bool) -> MLXArray {
        policy.setTrainingMode(false)
        let action = policy.forward(
            obs: observation,
            deterministic: deterministic,
            key: nil
        ).actions
        let envAction = actionForEnvironment(action)
        eval(envAction)
        return envAction
    }

    /// Signals the training loop to stop at the next step boundary.
    public func stop() {
        shouldContinue = false
    }

    public var numTimesteps: Int { timesteps }
    public var nUpdates: Int { updates }

    func policyParameterValues() -> [Float] {
        let arrays = policy.parameters().flattened().sorted { $0.0 < $1.0 }.map(\.1)
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
        updates: Int
    ) {
        self.timesteps = timesteps
        self.totalTimesteps = totalTimesteps
        self.progressRemaining = progressRemaining
        self.updates = updates
    }

    func train(
        rolloutBuffer: RolloutBuffer,
        callbacks: LearnCallbacks?
    ) async {
        guard rolloutBuffer.count > 0 else { return }

        policy.setTrainingMode(true)
        defer { policy.setTrainingMode(false) }
        let lr = Float(learningRate.value(at: progressRemaining))
        optimizer.learningRate = lr
        let step = buildTrainStep()

        guard let rollout = rolloutBuffer.prepared() else { return }
        var metricRows: [MLXArray] = []
        var continueTraining = true
        for _ in 0..<config.nEpochs {
            for batch in rollout.batches(batchSize: config.batchSize, generator: &batchGenerator) {
                let values = step(
                    [
                        batch.observations, batch.actions, batch.values,
                        batch.logProbs, batch.advantages, batch.returns,
                    ]
                )
                let metrics = MLX.stacked(values)
                if config.targetKL != nil {
                    eval(metrics, values[4], policy, optimizer)
                } else {
                    eval(metrics, policy, optimizer)
                }
                metricRows.append(metrics)

                if let targetKL = config.targetKL,
                    PPOMath.shouldStop(
                        approxKL: values[4].scalarValue(Float.self), targetKL: targetKL
                    )
                {
                    continueTraining = false
                    break
                }
            }
            updates += 1
            if !continueTraining { break }
        }
        guard let onTrain = callbacks?.onTrain else { return }
        let means = MLX.mean(MLX.stacked(metricRows), axis: 0).asArray(Float.self)

        let bufferStats = (
            values: rollout.batch.values.asArray(Float.self),
            returns: rollout.batch.returns.asArray(Float.self)
        )
        let explainedVariance = Self.explainedVariance(
            values: bufferStats.values,
            returns: bufferStats.returns
        )

        let metrics: [String: Double] = [
            LogKey.Train.loss: Double(means[0]),
            LogKey.Train.policyLoss: Double(means[1]),
            LogKey.Train.valueLoss: Double(means[2]),
            LogKey.Train.entropyLoss: Double(means[3]),
            LogKey.Train.learningRate: Double(lr),
            LogKey.Train.approxKL: Double(means[4]),
            LogKey.Train.clipFraction: Double(means[5]),
            LogKey.Train.nUpdates: Double(updates),
            "train/explained_variance": explainedVariance,
        ]

        await onTrain(metrics)
    }

    private func buildTrainStep() -> ([MLXArray]) -> [MLXArray] {
        if let step = compiledTrainStep {
            return step
        }

        let p = policy
        let opt = optimizer
        let clipRange = Float(config.clipRange)
        let clipRangeVf = config.clipRangeVf.map(Float.init)
        let entCoef = Float(config.entCoef)
        let vfCoef = Float(config.vfCoef)
        let maxGradNorm = Float(config.maxGradNorm)
        let normalizeAdvantage = config.normalizeAdvantage

        let vg = valueAndGrad(model: p) {
            (model: PPOPolicy, args: [MLXArray]) -> [MLXArray] in
            let observations = args[0]
            let actions = args[1]
            let oldValues = args[2]
            let oldLogProbs = args[3]
            let advantages =
                normalizeAdvantage && args[4].size > 1
                ? Self.normalizeAdvantages(args[4]) : args[4]
            let returns = args[5]

            let (valuesRaw, logProbRaw, entropyRaw) = model.evaluateActions(
                obs: observations,
                actions: actions
            )
            let values = Self.flatten(valuesRaw)
            let logProb = Self.flatten(logProbRaw)
            let losses = PPOMath.losses(
                values: values,
                logProb: logProb,
                entropy: entropyRaw.map(Self.flatten),
                oldValues: oldValues,
                oldLogProbs: oldLogProbs,
                advantages: advantages,
                returns: returns,
                clipRange: clipRange,
                clipRangeVf: clipRangeVf,
                entCoef: entCoef,
                vfCoef: vfCoef
            )
            return [losses[0]] + losses.dropFirst().map { MLX.stopGradient($0) }
        }

        let stepBody: ([MLXArray]) -> [MLXArray] = { (args: [MLXArray]) -> [MLXArray] in
            var (values, gradients) = vg(p, args)
            if maxGradNorm > 0.0 {
                gradients = Self.clipGradients(gradients, maxNorm: maxGradNorm)
            }
            opt.update(model: p, gradients: gradients)
            return values
        }

        let step: ([MLXArray]) -> [MLXArray]
        if p.useSDE
            || !supportsCompilation(p.piFeatureExtractor)
            || !supportsCompilation(p.vfFeatureExtractor)
        {
            step = stepBody
        } else {
            step = compile(
                inputs: [p, opt],
                outputs: [p, opt],
                stepBody
            )
        }

        compiledTrainStep = step
        return step
    }

    private func actionForEnvironment(_ action: MLXArray) -> MLXArray {
        if let box = boxSpace(from: policy.actionSpace) {
            var clipped = action.asType(.float32)
            if policy.squashOutput {
                clipped = (try? policy.unscaleAction(clipped)) ?? clipped
            }
            clipped = MLX.clip(clipped, min: box.low, max: box.high)
            return clipped
        }

        if policy.actionSpace is Discrete {
            var discreteAction = action.asType(.int32, stream: .cpu)
            if discreteAction.ndim == 0 {
                discreteAction = discreteAction.reshaped([1])
            } else if discreteAction.ndim > 1 {
                discreteAction = discreteAction.reshaped([-1])
                if discreteAction.shape[0] != 1 {
                    discreteAction = discreteAction[0].reshaped([1])
                }
            }
            return discreteAction
        }

        if let multiDiscrete = policy.actionSpace as? MultiDiscrete {
            var multiAction = action.asType(.int32, stream: .cpu).reshaped([-1])
            let high = multiDiscrete.nvec.asType(.int32).reshaped([-1]) - 1
            let low = MLX.zeros(high.shape, dtype: .int32)
            multiAction = MLX.clip(multiAction, min: low, max: high)
            return multiAction
        }

        if policy.actionSpace is MultiBinary {
            let binaryAction = (action.asType(.float32) .>= 0.5).asType(.int32)
            return binaryAction.reshaped([-1])
        }

        return action
    }

    func actionValues(observation: [Float], deterministic: Bool) -> [Float] {
        switch act(observation: observation, deterministic: deterministic) {
        case .discrete(let action):
            return [Float(action)]
        case .multiDiscrete(let actions):
            return actions.map(Float.init)
        case .multiBinary(let actions):
            return actions.map { $0 ? 1 : 0 }
        case .continuous(let actions):
            return actions
        }
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

    private static func flatten(_ value: MLXArray) -> MLXArray {
        if value.ndim == 0 {
            return value.reshaped([1])
        }
        return value.reshaped([-1])
    }

    private static func normalizeAdvantages(_ advantages: MLXArray) -> MLXArray {
        let mean = MLX.mean(advantages)
        let centered = advantages - mean
        let variance = MLX.mean(centered * centered)
        let std = MLX.sqrt(variance)
        return centered / (std + 1e-8)
    }

    private static func clipGradients(_ gradients: ModuleParameters, maxNorm: Float)
        -> ModuleParameters
    {
        let flattened = Dictionary(uniqueKeysWithValues: gradients.flattened())
        var totalNormSq = MLXArray(0.0)
        for (_, gradient) in flattened.sorted(by: { $0.key < $1.key }) {
            totalNormSq = totalNormSq + MLX.sum(gradient ** 2)
        }
        let totalNorm = MLX.sqrt(totalNormSq)
        let clipCoef = MLX.minimum(maxNorm / (totalNorm + 1e-6), 1.0)
        var clipped: [String: MLXArray] = [:]
        for (key, gradient) in flattened {
            clipped[key] = gradient * clipCoef
        }
        return ModuleParameters.unflattened(clipped)
    }

    static func explainedVariance(values: [Float], returns: [Float]) -> Double {
        guard values.count == returns.count, values.count > 1 else { return 0.0 }
        let n = Double(returns.count)
        let meanReturns = returns.reduce(0.0) { $0 + Double($1) } / n
        let varianceReturns =
            returns.reduce(0.0) { partial, value in
                let centered = Double(value) - meanReturns
                return partial + centered * centered
            } / n
        if !varianceReturns.isFinite || varianceReturns < 1e-12 {
            return 0.0
        }

        let residuals = zip(returns, values).map { Double($0.0) - Double($0.1) }
        let meanResidual = residuals.reduce(0.0, +) / n
        let varianceResidual =
            residuals.reduce(0.0) { partial, residual in
                let centered = residual - meanResidual
                return partial + centered * centered
            } / n
        let explained = 1.0 - varianceResidual / varianceReturns
        if explained.isFinite {
            return explained
        }
        return 0.0
    }
}
