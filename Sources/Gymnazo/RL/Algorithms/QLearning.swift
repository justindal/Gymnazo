//
//  QLearning.swift
//  Gymnazo
//

import MLX

/// Tabular Q-Learning algorithm for discrete state and action spaces.
///
/// Q-Learning is a model-free, off-policy reinforcement learning algorithm
/// that learns the optimal action-value function using a Q-table.
///
/// The update rule is:
/// ```
/// Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
/// ```
public final class QLearning: TabularTDAlgorithm {
    public let config: TabularTDConfig
    public var env: (any Env)?

    public let nStates: Int
    public let nActions: Int
    public let actionSpace: Discrete
    public var qTable: MLXArray
    public var explorationRate: Double
    public var randomKey: MLXArray

    public var numTimesteps: Int = 0
    public var totalTimesteps: Int = 0

    public var requiresNextAction: Bool { false }

    /// Creates a Q-Learning agent.
    ///
    /// - Parameters:
    ///   - nStates: Number of discrete states.
    ///   - nActions: Number of discrete actions.
    ///   - env: The environment to learn from.
    ///   - config: Hyperparameters.
    ///   - seed: Random seed for reproducibility.
    public init(
        nStates: Int,
        nActions: Int,
        env: (any Env)? = nil,
        config: TabularTDConfig = TabularTDConfig(),
        seed: UInt64? = nil
    ) {
        self.nStates = nStates
        self.nActions = nActions
        self.actionSpace = Discrete(n: nActions)
        self.env = env
        self.config = config
        self.qTable = MLX.zeros([nStates, nActions])
        self.explorationRate = config.explorationInitialEps
        self.randomKey = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
    }

    /// Creates a Q-Learning agent from an environment.
    ///
    /// - Parameters:
    ///   - env: The environment (must have Discrete observation and action spaces).
    ///   - config: Hyperparameters.
    ///   - seed: Random seed for reproducibility.
    public convenience init(
        env: any Env,
        config: TabularTDConfig = TabularTDConfig(),
        seed: UInt64? = nil
    ) {
        guard let obsSpace = env.observationSpace as? Discrete else {
            preconditionFailure("QLearning requires a Discrete observation space")
        }
        guard let actSpace = env.actionSpace as? Discrete else {
            preconditionFailure("QLearning requires a Discrete action space")
        }
        self.init(nStates: obsSpace.n, nActions: actSpace.n, env: env, config: config, seed: seed)
    }

    public func nextQ(nextState: MLXArray, nextAction: MLXArray?) -> MLXArray {
        let qValues = getQValues(for: nextState)
        return MLX.max(qValues)
    }
}
