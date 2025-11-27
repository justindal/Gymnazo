//
//  DeepRLAgent.swift
//

import Foundation
import MLX

/// Protocol for discrete action space deep RL agents (DQN, etc.)
public protocol DiscreteDeepRLAgent: AnyObject {
    var epsilon: Float { get set }
    
    func chooseAction(
        state: MLXArray,
        actionSpace: Discrete,
        key: inout MLXArray
    ) -> MLXArray
    
    func store(
        state: MLXArray,
        action: MLXArray,
        reward: Float,
        nextState: MLXArray,
        terminated: Bool
    )
    
    @discardableResult
    func update() -> (loss: Float, meanQ: Float, gradNorm: Float, tdError: Float)?
}

/// Protocol for continuous action space deep RL agents (SAC, etc.)
public protocol ContinuousDeepRLAgent: AnyObject {
    func chooseAction(
        state: MLXArray,
        key: inout MLXArray,
        deterministic: Bool
    ) -> MLXArray
    
    func store(
        state: MLXArray,
        action: MLXArray,
        reward: Float,
        nextState: MLXArray,
        terminated: Bool
    )
    
    @discardableResult
    func update() -> (qLoss: Float, actorLoss: Float, alphaLoss: Float)?
}

public typealias DeepRLAgent = DiscreteDeepRLAgent

