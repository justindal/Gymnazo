//
//  DQNNetworks.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Container for DQN networks (Q-network and target network).
public struct DQNNetworks {
    public let qNet: DQNPolicy
    public let qNetTarget: DQNPolicy

    /// Creates DQN networks from an existing policy.
    ///
    /// - Parameters:
    ///   - qNet: The Q-network policy.
    ///   - qNetTarget: The target Q-network policy.
    public init(qNet: DQNPolicy, qNetTarget: DQNPolicy) {
        self.qNet = qNet
        self.qNetTarget = qNetTarget
    }

    /// Synchronizes the target network parameters from the Q-network.
    public func syncTargetFromQNet() {
        _ = try? qNetTarget.update(parameters: qNet.parameters(), verify: .noUnusedKeys)
        qNetTarget.train(false)
    }

    /// Creates DQN networks from observation space and action count.
    ///
    /// - Parameters:
    ///   - observationSpace: The observation space.
    ///   - nActions: Number of discrete actions.
    ///   - config: Configuration for the Q-network.
    public init(
        observationSpace: any Space<MLXArray>,
        nActions: Int,
        config: DQNPolicyConfig = DQNPolicyConfig()
    ) {
        self.qNet = DQNPolicy(
            observationSpace: observationSpace,
            nActions: nActions,
            config: config
        )

        self.qNetTarget = DQNPolicy(
            observationSpace: observationSpace,
            nActions: nActions,
            config: config
        )

        syncTargetFromQNet()
    }

    /// Creates DQN networks from observation space and Discrete action space.
    ///
    /// - Parameters:
    ///   - observationSpace: The observation space.
    ///   - actionSpace: The discrete action space.
    ///   - config: Configuration for the Q-network.
    public init(
        observationSpace: any Space<MLXArray>,
        actionSpace: Discrete,
        config: DQNPolicyConfig = DQNPolicyConfig()
    ) {
        self.init(
            observationSpace: observationSpace,
            nActions: actionSpace.n,
            config: config
        )
    }

    /// Creates DQN networks from an existing Q-network.
    ///
    /// - Parameter qNet: The Q-network to use and create a target from.
    public init(qNet: DQNPolicy) {
        self.qNet = qNet

        self.qNetTarget = DQNPolicy(
            observationSpace: qNet.observationSpace,
            nActions: qNet.nActions,
            netArch: qNet.netArch,
            featuresExtractor: nil,
            normalizeImages: qNet.normalizeImages
        )

        syncTargetFromQNet()
    }
}
