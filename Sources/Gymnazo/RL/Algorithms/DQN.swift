//
//  DQN.swift
//  Gymnazo
//

import MLX
import MLXNN
import MLXOptimizers

/// Deep Q-Network (DQN) algorithm for discrete action spaces.
///
/// Implements vanilla DQN with experience replay and target network.
/// Uses epsilon-greedy exploration during training.
public final class DQN: OffPolicyAlgorithm, @unchecked Sendable {
    public typealias PolicyType = DQNPolicy

    public var config: OffPolicyConfig
    public let dqnConfig: DQNConfig
    public var replayBuffer: ReplayBuffer<MLXArray>?

    public let policy: DQNPolicy
    public let qNetTarget: DQNPolicy
    public var env: (any Env)?

    public var learningRate: any LearningRateSchedule
    public var currentProgressRemaining: Double
    public var numTimesteps: Int
    public var totalTimesteps: Int
    private var numGradientSteps: Int = 0

    public var optimizer: Adam

    private var explorationRate: Double
    private var randomKey: MLXArray
    private let seed: UInt64?

    public var qNet: DQNPolicy { policy }

    /// Creates a DQN from a concrete environment.
    ///
    /// - Parameters:
    ///   - env: The environment (must have Discrete action space).
    ///   - learningRate: Learning rate schedule.
    ///   - policyConfig: Configuration for the Q-network.
    ///   - config: DQN hyperparameters.
    ///   - optimizerConfig: Optimizer configuration.
    ///   - seed: Random seed for reproducibility.
    public convenience init(
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

        self.init(
            observationSpace: env.observationSpace,
            actionSpace: discrete,
            env: env,
            learningRate: learningRate,
            policyConfig: policyConfig,
            config: config,
            optimizerConfig: optimizerConfig,
            seed: seed
        )
    }

    /// Creates a DQN from pre-built networks.
    ///
    /// - Parameters:
    ///   - networks: The Q-network and target network.
    ///   - env: Optional environment.
    ///   - learningRate: Learning rate schedule.
    ///   - config: DQN hyperparameters.
    ///   - optimizerConfig: Optimizer configuration.
    ///   - seed: Random seed.
    public init(
        networks: DQNNetworks,
        env: (any Env)? = nil,
        learningRate: any LearningRateSchedule = ConstantLearningRate(1e-4),
        config: DQNConfig = DQNConfig(),
        optimizerConfig: DQNOptimizerConfig = DQNOptimizerConfig(),
        seed: UInt64? = nil
    ) {
        self.randomKey = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
        self.policy = networks.qNet
        self.qNetTarget = networks.qNetTarget
        self.env = env
        self.learningRate = learningRate
        self.dqnConfig = config
        self.config = DQN.makeOffPolicyConfig(from: config)
        self.currentProgressRemaining = 1.0
        self.numTimesteps = 0
        self.totalTimesteps = 0
        self.explorationRate = config.explorationInitialEps
        self.seed = seed

        let lr = Float(learningRate.value(at: 1.0))
        self.optimizer = optimizerConfig.optimizer.make(learningRate: lr)
    }

    /// Creates a DQN from an existing Q-network.
    ///
    /// - Parameters:
    ///   - policy: The Q-network policy.
    ///   - env: Optional environment.
    ///   - learningRate: Learning rate schedule.
    ///   - config: DQN hyperparameters.
    ///   - optimizerConfig: Optimizer configuration.
    ///   - seed: Random seed.
    public convenience init(
        policy: DQNPolicy,
        env: (any Env)? = nil,
        learningRate: any LearningRateSchedule = ConstantLearningRate(1e-4),
        config: DQNConfig = DQNConfig(),
        optimizerConfig: DQNOptimizerConfig = DQNOptimizerConfig(),
        seed: UInt64? = nil
    ) {
        let networks = DQNNetworks(qNet: policy)
        self.init(
            networks: networks,
            env: env,
            learningRate: learningRate,
            config: config,
            optimizerConfig: optimizerConfig,
            seed: seed
        )
    }

    /// Creates a DQN from observation and action spaces.
    ///
    /// - Parameters:
    ///   - observationSpace: The observation space.
    ///   - actionSpace: The discrete action space.
    ///   - env: Optional environment.
    ///   - learningRate: Learning rate schedule.
    ///   - policyConfig: Configuration for the Q-network.
    ///   - config: DQN hyperparameters.
    ///   - optimizerConfig: Optimizer configuration.
    ///   - seed: Random seed.
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

        self.randomKey = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
        self.policy = networks.qNet
        self.qNetTarget = networks.qNetTarget
        self.env = env
        self.learningRate = learningRate
        self.dqnConfig = config
        self.config = DQN.makeOffPolicyConfig(from: config)
        self.currentProgressRemaining = 1.0
        self.numTimesteps = 0
        self.totalTimesteps = 0
        self.explorationRate = config.explorationInitialEps
        self.seed = seed

        let lr = Float(learningRate.value(at: 1.0))
        self.optimizer = optimizerConfig.optimizer.make(learningRate: lr)
    }

    public func setupReplayBuffer() {
        guard replayBuffer == nil else { return }
        let bufferConfig = ReplayBuffer<MLXArray>.Configuration(
            bufferSize: dqnConfig.bufferSize,
            optimizeMemoryUsage: dqnConfig.optimizeMemoryUsage,
            handleTimeoutTermination: dqnConfig.handleTimeoutTermination,
            seed: seed
        )
        replayBuffer = ReplayBuffer(
            observationSpace: policy.observationSpace,
            actionSpace: policy.actionSpace,
            config: bufferConfig,
            numEnvs: 1
        )
    }

    /// Trains the DQN for the specified number of timesteps.
    ///
    /// - Parameter totalTimesteps: Total environment steps to train for.
    /// - Returns: Self for chaining.
    @discardableResult
    public func learn(totalTimesteps: Int) throws -> Self {
        try learn(totalTimesteps: totalTimesteps, callbacks: nil)
    }

    /// Trains the DQN for the specified number of timesteps with optional callbacks.
    ///
    /// - Parameters:
    ///   - totalTimesteps: Total environment steps to train for.
    ///   - callbacks: Optional callbacks for step updates and episode completion.
    /// - Returns: Self for chaining.
    @discardableResult
    public func learn(totalTimesteps: Int, callbacks: LearnCallbacks?) throws -> Self {
        self.totalTimesteps = totalTimesteps
        self.numTimesteps = 0

        setupReplayBuffer()

        guard var environment = env else {
            throw GymnazoError.invalidState(
                "DQN.learn requires an environment. Set env before calling learn()."
            )
        }

        var lastObs = try environment.reset().obs
        var numCollectedSteps = 0
        var numCollectedEpisodes = 0
        var stepsSinceLastTrain = 0
        var episodeReward: Double = 0
        var episodeLength: Int = 0

        while numTimesteps < totalTimesteps {
            updateExplorationRate()

            let action: MLXArray

            let (exploreKey, nextKey) = MLX.split(key: randomKey)
            randomKey = nextKey

            if shouldExplore(key: exploreKey) {
                let (sampleKey, nextKey2) = MLX.split(key: randomKey)
                randomKey = nextKey2
                action = sampleRandomAction(key: sampleKey)
            } else {
                action = selectAction(obs: lastObs)
            }

            let stepResult = try environment.step(action)
            let reward = Float(stepResult.reward)
            let terminated = stepResult.terminated
            let truncated = stepResult.truncated

            episodeReward += Double(reward)
            episodeLength += 1

            let bufferNextObs =
                stepResult.info["final_observation"]?.cast(MLXArray.self) ?? stepResult.obs

            storeTransition(
                obs: lastObs,
                action: action,
                reward: reward,
                nextObs: bufferNextObs,
                terminated: terminated,
                truncated: truncated
            )

            eval(action, bufferNextObs)

            if terminated || truncated {
                callbacks?.onEpisodeEnd?(episodeReward, episodeLength)
                episodeReward = 0
                episodeLength = 0

                lastObs = try environment.reset().obs
                numCollectedEpisodes += 1
            } else {
                lastObs = stepResult.obs
            }

            numTimesteps += 1
            numCollectedSteps += 1
            let isWarmup = numTimesteps < dqnConfig.learningStarts
            if !isWarmup {
                stepsSinceLastTrain += 1
            }
            currentProgressRemaining = 1.0 - Double(numTimesteps) / Double(totalTimesteps)

            if let onStep = callbacks?.onStep {
                let shouldContinue = onStep(numTimesteps, totalTimesteps, explorationRate)
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
            switch dqnConfig.trainFrequency.unit {
            case .step:
                shouldTrain = numCollectedSteps % dqnConfig.trainFrequency.frequency == 0
            case .episode:
                shouldTrain =
                    (terminated || truncated)
                    && numCollectedEpisodes % dqnConfig.trainFrequency.frequency == 0
            }

            if shouldTrain && numTimesteps >= dqnConfig.learningStarts {
                let gradientSteps: Int
                switch dqnConfig.gradientSteps {
                case .fixed(let steps):
                    gradientSteps = steps
                case .asCollectedSteps:
                    gradientSteps = stepsSinceLastTrain
                }
                train(gradientSteps: gradientSteps, batchSize: dqnConfig.batchSize)
                stepsSinceLastTrain = 0
            }
        }

        self.env = environment
        return self
    }

    private func updateExplorationRate() {
        let fraction = min(
            1.0, Double(numTimesteps) / (Double(totalTimesteps) * dqnConfig.explorationFraction))
        explorationRate =
            dqnConfig.explorationInitialEps
            + fraction * (dqnConfig.explorationFinalEps - dqnConfig.explorationInitialEps)
    }

    private func shouldExplore(key: MLXArray) -> Bool {
        let random = MLX.uniform(0.0..<1.0, key: key)
        eval(random)
        let value: Float = random.item()
        return Double(value) < explorationRate
    }

    private func sampleRandomAction(key: MLXArray) -> MLXArray {
        guard let discrete = env?.actionSpace as? Discrete else {
            preconditionFailure("DQN requires a Discrete action space")
        }
        return discrete.sample(key: key, mask: nil, probability: nil)
    }

    private func selectAction(obs: MLXArray) -> MLXArray {
        policy.setTrainingMode(false)
        let qValues = policy.forward(obs: obs)
        var action = MLX.argMax(qValues, axis: -1).asType(.int32)
        if action.ndim == 0 {
            action = action.reshaped([1])
        }
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

    public func train(gradientSteps: Int, batchSize: Int) {
        guard var buffer = replayBuffer, buffer.count >= batchSize else { return }

        let lr = Float(learningRate.value(at: currentProgressRemaining))
        optimizer.learningRate = lr

        for _ in 0..<gradientSteps {
            let (sampleKey, nextKey) = MLX.split(key: randomKey)
            randomKey = nextKey
            let batch = buffer.sample(batchSize, key: sampleKey)
            trainStep(batch: batch)
        }
        replayBuffer = buffer
    }

    private func trainStep(batch: ReplayBuffer<MLXArray>.Sample) {
        qNetTarget.setTrainingMode(false)

        let gamma = Float(dqnConfig.gamma)

        let targetQValues = targetQ(
            nextObs: batch.nextObs,
            rewards: batch.rewards,
            dones: batch.dones,
            gamma: gamma
        )
        eval(targetQValues)
        let detachedTargetQ = MLX.stopGradient(targetQValues)

        policy.setTrainingMode(true)

        typealias QNetArgs = (obs: MLXArray, actions: MLXArray, targetQ: MLXArray)
        let qNetVG = valueAndGrad(model: policy) {
            (model: DQNPolicy, args: QNetArgs) -> [MLXArray] in
            [self.qNetLoss(model, obs: args.obs, actions: args.actions, targetQ: args.targetQ)]
        }

        var (_, grads) = qNetVG(policy, (batch.obs, batch.actions, detachedTargetQ))

        if let maxNorm = dqnConfig.maxGradNorm {
            grads = clipGradients(grads, maxNorm: Float(maxNorm))
        }

        optimizer.update(model: policy, gradients: grads)
        eval(policy.parameters())

        numGradientSteps += 1
        if numGradientSteps % dqnConfig.targetUpdateInterval == 0 {
            updateTarget()
        }
    }

    private func targetQ(
        nextObs: MLXArray,
        rewards: MLXArray,
        dones: MLXArray,
        gamma: Float
    ) -> MLXArray {
        let nextQValues = qNetTarget.forward(obs: nextObs)
        let nextQMax = MLX.max(nextQValues, axis: -1, keepDims: true)

        return rewards.expandedDimensions(axis: -1)
            + (1.0 - dones.expandedDimensions(axis: -1)) * gamma * nextQMax
    }

    private func qNetLoss(
        _ qNet: DQNPolicy,
        obs: MLXArray,
        actions: MLXArray,
        targetQ: MLXArray
    ) -> MLXArray {
        let qValues = qNet.forward(obs: obs)

        let batchSize = actions.shape[0]
        let batchIndices = MLXArray(Array(0..<Int32(batchSize)))
        let actionIndices = actions.reshaped([-1]).asType(.int32)

        let selectedQ = qValues[batchIndices, actionIndices].expandedDimensions(axis: -1)

        return MLX.mean((selectedQ - targetQ) ** 2)
    }

    /// Updates the target network using Polyak averaging.
    public func updateTarget() {
        updateTarget(tau: dqnConfig.tau)
    }

    /// Updates the target network with a custom tau value.
    ///
    /// - Parameter tau: Interpolation factor. 1.0 = hard update, < 1.0 = soft update.
    public func updateTarget(tau: Double) {
        let qNetParams = policy.parameters()
        let targetParams = qNetTarget.parameters()

        let updated = polyakUpdate(target: targetParams, source: qNetParams, tau: tau)
        _ = try? qNetTarget.update(parameters: updated, verify: .noUnusedKeys)
        qNetTarget.setTrainingMode(false)
        eval(qNetTarget.parameters())
    }

    /// Gets the current exploration rate.
    public var currentExplorationRate: Double {
        explorationRate
    }

    /// Predicts the action for a given observation using epsilon-greedy.
    ///
    /// - Parameters:
    ///   - observation: The observation.
    ///   - deterministic: If true, ignores exploration and returns greedy action.
    /// - Returns: The selected action as MLXArray.
    public func predict(observation: MLXArray, deterministic: Bool = false) -> MLXArray {
        if !deterministic {
            let (exploreKey, nextKey) = MLX.split(key: randomKey)
            randomKey = nextKey

            if shouldExplore(key: exploreKey) {
                let (sampleKey, nextKey2) = MLX.split(key: randomKey)
                randomKey = nextKey2
                return sampleRandomAction(key: sampleKey)
            }
        }

        return selectAction(obs: observation)
    }

    private func clipGradients(_ gradients: ModuleParameters, maxNorm: Float) -> ModuleParameters {
        let flattened = Dictionary(uniqueKeysWithValues: gradients.flattened())

        var totalNormSq = MLXArray(Float(0.0))
        for (_, grad) in flattened {
            totalNormSq = totalNormSq + MLX.sum(grad ** 2)
        }
        let totalNorm = MLX.sqrt(totalNormSq)

        let clipCoef = MLX.minimum(MLXArray(maxNorm) / (totalNorm + 1e-6), MLXArray(Float(1.0)))

        var clipped: [String: MLXArray] = [:]
        for (key, grad) in flattened {
            clipped[key] = grad * clipCoef
        }

        return ModuleParameters.unflattened(clipped)
    }

    private static func makeOffPolicyConfig(from dqnConfig: DQNConfig) -> OffPolicyConfig {
        OffPolicyConfig(
            bufferSize: dqnConfig.bufferSize,
            learningStarts: dqnConfig.learningStarts,
            batchSize: dqnConfig.batchSize,
            tau: dqnConfig.tau,
            gamma: dqnConfig.gamma,
            trainFrequency: dqnConfig.trainFrequency,
            gradientSteps: dqnConfig.gradientSteps,
            targetUpdateInterval: dqnConfig.targetUpdateInterval,
            optimizeMemoryUsage: dqnConfig.optimizeMemoryUsage,
            handleTimeoutTermination: dqnConfig.handleTimeoutTermination,
            useSDEAtWarmup: false,
            sdeSampleFreq: -1,
            sdeSupported: false
        )
    }
}
