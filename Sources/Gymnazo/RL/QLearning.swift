import Foundation
import MLX

/// Minimal Q-Learning agent storing a tabular value function backed by MLXArray.
public final class QLearningAgent: DiscreteRLAgent {
    public var learningRate: Float
    public var gamma: Float
    public let stateSize: Int
    public let actionSize: Int
    public var epsilon: Float
    public private(set) var qTable: MLXArray

    public init(learningRate: Float, gamma: Float, stateSize: Int, actionSize: Int, epsilon: Float) {
        self.learningRate = learningRate
        self.gamma = gamma
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.epsilon = epsilon
        self.qTable = MLXArray.zeros([stateSize, actionSize], type: Float.self)
    }

    /// Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
    @discardableResult
    public func update(
        state: Int,
        action: Int,
        reward: Float,
        nextState: Int,
        nextAction: Int,
        terminated: Bool
    ) -> (newQ: Float, tdError: Float) {
        precondition(state >= 0 && state < stateSize, "state out of bounds")
        precondition(action >= 0 && action < actionSize, "action out of bounds")
        precondition(nextState >= 0 && nextState < stateSize, "next state out of bounds")

        let target: MLXArray
        if terminated {
            target = MLXArray(reward)
        } else {
            // target = MLXArray(reward) + MLXArray(gamma) * qTable[nextState, 0...].max()
            target = MLXArray(reward + gamma * qTable[nextState, 0...].max().item(Float.self))
        }

        let currentValue = qTable[state, action]
        let tdError = target - currentValue
        let updatedValue = currentValue + MLXArray(learningRate) * tdError
        qTable[state, action] = updatedValue
        return (updatedValue.item(Float.self), tdError.item(Float.self))
    }

    public func resetTable() {
        self.qTable = MLXArray.zeros([stateSize, actionSize], type: Float.self)
    }
    
    /// Load a pre-trained Q-table
    public func loadQTable(_ table: MLXArray) {
        precondition(table.shape == [stateSize, actionSize], 
                     "Q-table shape mismatch: expected [\(stateSize), \(actionSize)], got \(table.shape)")
        self.qTable = table
    }

    /// epsilon-greedy action selection supporting discrete action spaces.
    public func chooseAction(
        actionSpace: Discrete,
        state: Int,
        key: inout MLXArray
    ) -> Int {
        precondition(state >= 0 && state < stateSize, "state out of bounds")

        let (newKey, rollKey) = MLX.split(key: key)
        key = newKey
        
        let (epsilonKey, actionKey) = MLX.split(key: rollKey)
        
        let roll = MLX.uniform(0 ..< 1, key: epsilonKey).item() as Float
        if roll < epsilon {
            return actionSpace.sample(key: actionKey)
        }

        let row = qTable[state, 0...]
        let maxVal = row.max()
        
        let mask = row .== maxVal
        
        let zero = MLXArray(0, dtype: .float32)
        let negInf = MLXArray(-Float.infinity, dtype: .float32)
        let logits = MLX.which(mask, zero, negInf)
        
        let choiceIndex = MLX.categorical(logits, key: actionKey)
        let chosen = choiceIndex.item(Int.self)
        return chosen + actionSpace.start
    }
}