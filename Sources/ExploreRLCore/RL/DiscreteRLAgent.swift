//
//  DiscreteRLAgent.swift
//

import Foundation
import MLX

/// A protocol defining the requirements for a reinforcement learning agent 
/// operating in discrete state and action spaces.
public protocol DiscreteRLAgent: AnyObject {
    var epsilon: Float { get set }
    var qTable: MLXArray { get }
    
    func resetTable()

    func chooseAction(
        actionSpace: Discrete,
        state: Int,
        key: inout MLXArray
    ) -> Int
    
    @discardableResult
    func update(
        state: Int,
        action: Int,
        reward: Float,
        nextState: Int,
        nextAction: Int,
        terminated: Bool
    ) -> (newQ: Float, tdError: Float)
}

public class DiscreteAgent: DiscreteRLAgent {
    private let _agent: any DiscreteRLAgent
    
    public init<T: DiscreteRLAgent>(_ agent: T) {
        self._agent = agent
    }
    
    public var qTable: MLXArray { _agent.qTable }
    
    public var epsilon: Float {
        get { _agent.epsilon }
        set {
            var mutableAgent = _agent
            mutableAgent.epsilon = newValue
        }
    }
    
    public func resetTable() {
        _agent.resetTable()
    }
    
    public func chooseAction(actionSpace: Discrete, state: Int, key: inout MLXArray) -> Int {
        _agent.chooseAction(actionSpace: actionSpace, state: state, key: &key)
    }
    
    @discardableResult
    public func update(
        state: Int,
        action: Int,
        reward: Float,
        nextState: Int,
        nextAction: Int,
        terminated: Bool
    ) -> (newQ: Float, tdError: Float) {
        _agent.update(
            state: state,
            action: action,
            reward: reward,
            nextState: nextState,
            nextAction: nextAction,
            terminated: terminated
        )
    }
}

