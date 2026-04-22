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

    nonisolated(unsafe) private var env: (any Env)?
    let policy: PPOPolicy
    private var optimizer: Adam
    let learningRate: any LearningRateSchedule
    let randomSeed: UInt64?

    var timesteps: Int = 0
    var totalTimesteps: Int = 0
    var updates: Int = 0
    var progressRemaining: Double = 1.0

    private var shouldContinue: Bool = true
    private var key: MLXArray
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
        env: any Env,
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
            useSDE: config.useSDE
        )
        self.learningRate = learningRate
        self.randomSeed = seed
        self.key = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))

        let optimizerConfig: OptimizerConfig = .adam()
        self.optimizer = optimizerConfig.make(
            learningRate: Float(learningRate.value(at: 1.0))
        )
    }

    init(
        policy: PPOPolicy,
        optimizer: Adam,
        config: PPOConfig,
        policyConfig: PPOPolicyConfig,
        learningRate: any LearningRateSchedule,
        seed: UInt64?,
        timesteps: Int,
        totalTimesteps: Int,
        progressRemaining: Double,
        updates: Int
    ) {
        self.env = nil
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

                var adjustedReward = Float(step.reward)
                if step.truncated,
                    let finalObservation = step.info["final_observation"]?.cast(MLXArray.self)
                {
                    let bootstrapValue = policy.predictValues(obs: finalObservation)
                    let terminalValue = Self.flatten(bootstrapValue).item(Float.self)
                    adjustedReward += Float(config.gamma) * terminalValue
                }

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
        guard var environment = env else {
            throw GymnazoError.invalidState("PPO.evaluate requires an environment.")
        }

        for _ in 0..<episodes {
            var obs = try environment.reset().obs
            var done = false
            var episodeReward: Double = 0.0
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
    /// - Parameters:
    ///   - observation: The current environment observation.
    ///   - deterministic: If `true` (default), returns the mode action.
    /// - Returns: The selected action as an `MLXArray` clipped to the action space.
    public func callAsFunction(
        observation: MLXArray,
        deterministic: Bool = true
    ) -> MLXArray {
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
        let environment = env
        env = nil
        return environment
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

    private func train(
        rolloutBuffer: RolloutBuffer,
        callbacks: LearnCallbacks?
    ) async {
        guard rolloutBuffer.count > 0 else { return }

        policy.setTrainingMode(true)
        let lr = Float(learningRate.value(at: progressRemaining))
        optimizer.learningRate = lr
        let step = buildTrainStep()

        var lossValues: [Float] = []
        var policyLossValues: [Float] = []
        var valueLossValues: [Float] = []
        var entropyLossValues: [Float] = []
        var approxKLValues: [Float] = []
        var clipFractionValues: [Float] = []
        lossValues.reserveCapacity(config.nEpochs)
        policyLossValues.reserveCapacity(config.nEpochs)
        valueLossValues.reserveCapacity(config.nEpochs)
        entropyLossValues.reserveCapacity(config.nEpochs)
        approxKLValues.reserveCapacity(config.nEpochs)
        clipFractionValues.reserveCapacity(config.nEpochs)

        var continueTraining = true
        for _ in 0..<config.nEpochs {
            let (batchKey, nextKey) = MLX.split(key: key, stream: .cpu)
            key = nextKey
            let batches = rolloutBuffer.batches(
                batchSize: config.batchSize,
                key: batchKey
            )
            for batch in batches {
                var advantages = batch.advantages
                if config.normalizeAdvantage && advantages.size > 1 {
                    advantages = Self.normalizeAdvantages(advantages)
                }
                let values = step(
                    [
                        batch.observations,
                        batch.actions,
                        batch.values,
                        batch.logProbs,
                        advantages,
                        batch.returns,
                    ]
                )
                eval(
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                    policy,
                    optimizer
                )

                let loss = values[0].item(Float.self)
                let policyLoss = values[1].item(Float.self)
                let valueLoss = values[2].item(Float.self)
                let entropyLoss = values[3].item(Float.self)
                let approxKL = values[4].item(Float.self)
                let clipFraction = values[5].item(Float.self)

                lossValues.append(loss)
                policyLossValues.append(policyLoss)
                valueLossValues.append(valueLoss)
                entropyLossValues.append(entropyLoss)
                approxKLValues.append(approxKL)
                clipFractionValues.append(clipFraction)

                if let targetKL = config.targetKL, approxKL > Float(1.5 * targetKL) {
                    continueTraining = false
                    break
                }
            }
            updates += 1
            if !continueTraining {
                break
            }
        }

        let bufferStats = rolloutBuffer.valuesAndReturns()
        let explainedVariance = Self.explainedVariance(
            values: bufferStats.values,
            returns: bufferStats.returns
        )

        let metrics: [String: Double] = [
            LogKey.Train.loss: Double(Self.mean(lossValues)),
            LogKey.Train.policyLoss: Double(Self.mean(policyLossValues)),
            LogKey.Train.valueLoss: Double(Self.mean(valueLossValues)),
            LogKey.Train.entropyLoss: Double(Self.mean(entropyLossValues)),
            LogKey.Train.learningRate: Double(lr),
            LogKey.Train.approxKL: Double(Self.mean(approxKLValues)),
            LogKey.Train.clipFraction: Double(Self.mean(clipFractionValues)),
            LogKey.Train.nUpdates: Double(updates),
            "train/explained_variance": explainedVariance,
        ]

        await callbacks?.onTrain?(metrics)
        policy.setTrainingMode(false)
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

        let vg = valueAndGrad(model: p) {
            (model: PPOPolicy, args: [MLXArray]) -> [MLXArray] in
            let observations = args[0]
            let actions = args[1]
            let oldValues = args[2]
            let oldLogProbs = args[3]
            let advantages = args[4]
            let returns = args[5]

            let (valuesRaw, logProbRaw, entropyRaw) = model.evaluateActions(
                obs: observations,
                actions: actions
            )
            let values = Self.flatten(valuesRaw)
            let logProb = Self.flatten(logProbRaw)
            let ratio = MLX.exp(logProb - oldLogProbs)

            let policyLoss1 = advantages * ratio
            let policyLoss2 =
                advantages
                * MLX.clip(
                    ratio,
                    min: 1.0 - clipRange,
                    max: 1.0 + clipRange
                )
            let policyLoss = -MLX.mean(MLX.minimum(policyLoss1, policyLoss2))

            let valuePred: MLXArray
            if let clipRangeVf {
                valuePred =
                    oldValues
                    + MLX.clip(
                        values - oldValues,
                        min: -clipRangeVf,
                        max: clipRangeVf
                    )
            } else {
                valuePred = values
            }
            let valueLoss = MLX.mean((returns - valuePred) ** 2)

            let entropyLoss: MLXArray
            if let entropyRaw {
                entropyLoss = -MLX.mean(Self.flatten(entropyRaw))
            } else {
                entropyLoss = MLX.mean(logProb)
            }

            let totalLoss = policyLoss + entCoef * entropyLoss + vfCoef * valueLoss

            let logRatio = logProb - oldLogProbs
            let approxKL = MLX.mean((MLX.exp(logRatio) - 1.0) - logRatio)
            let clipFraction = MLX.mean((MLX.abs(ratio - 1.0) .> clipRange).asType(.float32))

            return [
                totalLoss,
                MLX.stopGradient(policyLoss),
                MLX.stopGradient(valueLoss),
                MLX.stopGradient(entropyLoss),
                MLX.stopGradient(approxKL),
                MLX.stopGradient(clipFraction),
            ]
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
            || !(p.piFeatureExtractor is FlattenExtractor)
            || !(p.vfFeatureExtractor is FlattenExtractor)
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
        for (_, gradient) in flattened {
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

    private static func explainedVariance(values: [Float], returns: [Float]) -> Double {
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

        let varianceResidual =
            zip(returns, values).reduce(0.0) { partial, pair in
                let residual = Double(pair.0 - pair.1)
                return partial + residual * residual
            } / n
        let explained = 1.0 - varianceResidual / varianceReturns
        if explained.isFinite {
            return explained
        }
        return 0.0
    }

    private static func mean(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0.0 }
        let total = values.reduce(0.0, +)
        return total / Float(values.count)
    }
}
