//
//  SARSA.swift
//  Gymnazo
//

import MLX

/// SARSA (State-Action-Reward-State-Action) algorithm for discrete state and action spaces.
///
/// SARSA is a model-free, on-policy reinforcement learning algorithm
/// that learns the action-value function based on the policy being followed.
///
/// The update rule is:
/// ```
/// Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
/// ```
///
/// Unlike Q-Learning, SARSA uses the actual next action (a') taken by the policy,
/// making it an on-policy algorithm that learns the value of the policy being followed.
public final class SARSA<Environment: Env>: TabularTDAlgorithm
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

    public var requiresNextAction: Bool { true }

    /// Creates a SARSA agent.
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

    /// Creates a SARSA agent from an environment.
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
            preconditionFailure("SARSA requires a Discrete action space")
        }
        self.init(nActions: discrete.n, env: env, config: config, seed: seed)
    }

    public func computeNextQ(nextState: Int, nextAction: Int?) -> Double {
        guard let action = nextAction else {
            return 0.0
        }
        return getQValue(state: nextState, action: action)
    }
}
