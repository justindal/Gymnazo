//
//  DeepRLAgent.swift
//

import Foundation
import MLX

/// A protocol defining the requirements for a Deep Reinforcement Learning agent.
public protocol DeepRLAgent: AnyObject {
    var epsilon: Float { get set }
    
    /// selects an action based on the current state (epsilon-greedy or policy-based).
    func chooseAction(
        state: MLXArray,
        actionSpace: Discrete,
        key: inout MLXArray
    ) -> MLXArray
    
    /// ssave experience in the replay buffer.
    func store(
        state: MLXArray,
        action: MLXArray,
        reward: Float,
        nextState: MLXArray,
        terminated: Bool
    )
    
    /// step by sampling from the buffer and updating the network.
    /// returns the loss value, mean Q-value, gradient norm, and TD error (nil if no update occurred or buffer too small)
    @discardableResult
    func update() -> (loss: Float, meanQ: Float, gradNorm: Float, tdError: Float)?
}

