import Foundation
import MLX
import MLXRandom

/// Minimal Q-Learning agent storing a tabular value function backed by MLXArray.
public struct QLearningAgent {
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
    public mutating func update(
        state: Int,
        action: Int,
        reward: Float,
        nextState: Int
    ) -> Float {
        precondition(state >= 0 && state < stateSize, "state out of bounds")
        precondition(action >= 0 && action < actionSize, "action out of bounds")
        precondition(nextState >= 0 && nextState < stateSize, "next state out of bounds")

        var currentRow = qTable[state, 0...].asArray(Float.self)
        let nextRow = qTable[nextState, 0...].asArray(Float.self)

        let currentValue = currentRow[action]
        let maxNext = nextRow.max() ?? 0
        let target = reward + gamma * maxNext
        let updatedValue = currentValue + learningRate * (target - currentValue)
        currentRow[action] = updatedValue

        qTable[state, 0...] = MLXArray(currentRow)
        return updatedValue
    }

    public mutating func resetTable() {
        self.qTable = MLXArray.zeros([stateSize, actionSize], type: Float.self)
    }

    /// epsilon-greedy action selection supporting discrete action spaces.
    public func chooseAction(
        actionSpace: Discrete,
        state: Int,
        key: inout MLXArray
    ) -> Int {
        precondition(state >= 0 && state < stateSize, "state out of bounds")

        let (newKey, rollKey) = MLXRandom.split(key: key)
        key = newKey
        let roll = MLXRandom.uniform(0 ..< 1, key: rollKey).item() as Float
        if roll < epsilon {
            return actionSpace.sample(key: rollKey)
        }

        let row: [Float] = qTable[state, 0...].asArray(Float.self)
        guard let maxValue: Float = row.max() else {
            return actionSpace.sample(key: rollKey)
        }

        let bestIndices: [Int] = row.enumerated().compactMap { idx, value -> Int? in
            abs(value - maxValue) <= Float.ulpOfOne ? idx : nil
        }

        guard !bestIndices.isEmpty else {
            return actionSpace.sample(key: rollKey)
        }

        let (choiceKey, _) = MLXRandom.split(key: key)
        key = choiceKey
        let choiceIndex: Int = MLXRandom.randInt(0 ..< bestIndices.count, key: choiceKey).item() as Int
        let chosen: Int = bestIndices[choiceIndex]
        return chosen + actionSpace.start
    }
}