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
public final class QLearning<Environment: Env>: TabularTDAlgorithm
where Environment.Observation == Int, Environment.Action == Int {
    public let config: TabularTDConfig
    public var env: Environment?

    public let nActions: Int
    public let actionSpace: Discrete
    public var qTable: [Int: [Double]]
    public var explorationRate: Double
    public var randomKey: MLXArray

    public var numTimesteps: Int = 0
    public var totalTimesteps: Int = 0

    public var requiresNextAction: Bool { false }

    /// Creates a Q-Learning agent.
    ///
    /// - Parameters:
    ///   - nActions: Number of discrete actions.
    ///   - env: The environment to learn from.
    ///   - config: Hyperparameters.
    ///   - seed: Random seed for reproducibility.
    public init(
        nActions: Int,
        env: Environment? = nil,
        config: TabularTDConfig = TabularTDConfig(),
        seed: UInt64? = nil
    ) {
        self.nActions = nActions
        self.actionSpace = Discrete(n: nActions)
        self.env = env
        self.config = config
        self.qTable = [:]
        self.explorationRate = config.explorationInitialEps
        self.randomKey = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
    }

    /// Creates a Q-Learning agent from an environment.
    ///
    /// - Parameters:
    ///   - env: The environment (must have Discrete action space).
    ///   - config: Hyperparameters.
    ///   - seed: Random seed for reproducibility.
    public convenience init(
        env: Environment,
        config: TabularTDConfig = TabularTDConfig(),
        seed: UInt64? = nil
    ) {
        guard let discrete = env.actionSpace as? Discrete else {
            preconditionFailure("QLearning requires a Discrete action space")
        }
        self.init(nActions: discrete.n, env: env, config: config, seed: seed)
    }

    public func computeNextQ(nextState: Int, nextAction: Int?) -> Double {
        getQValues(for: nextState).max() ?? 0.0
    }
}
