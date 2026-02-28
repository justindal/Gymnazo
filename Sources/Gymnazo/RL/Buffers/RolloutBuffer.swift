import MLX

public struct RolloutBuffer: RolloutBuffering {
    public let bufferSize: Int
    public let observationSpace: any Space
    public let actionSpace: any Space
    public let numEnvs: Int

    public private(set) var count: Int = 0

    private let actionDim: Int
    private var observations: [MLXArray]
    private var actions: [MLXArray]
    private var rewards: [Float]
    private var episodeStarts: [Float]
    private var values: [Float]
    private var logProbs: [Float]
    private var advantages: [Float]
    private var returns: [Float]

    public init(
        bufferSize: Int,
        observationSpace: any Space,
        actionSpace: any Space,
        numEnvs: Int = 1
    ) {
        precondition(bufferSize > 0, "RolloutBuffer requires bufferSize > 0")
        precondition(numEnvs == 1, "RolloutBuffer currently supports only numEnvs == 1")
        self.bufferSize = bufferSize
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        self.numEnvs = numEnvs
        self.actionDim = Self.actionDimension(actionSpace)
        self.observations = []
        self.actions = []
        self.rewards = []
        self.episodeStarts = []
        self.values = []
        self.logProbs = []
        self.advantages = []
        self.returns = []
        self.observations.reserveCapacity(bufferSize)
        self.actions.reserveCapacity(bufferSize)
        self.rewards.reserveCapacity(bufferSize)
        self.episodeStarts.reserveCapacity(bufferSize)
        self.values.reserveCapacity(bufferSize)
        self.logProbs.reserveCapacity(bufferSize)
        self.advantages.reserveCapacity(bufferSize)
        self.returns.reserveCapacity(bufferSize)
    }

    public mutating func reset() {
        count = 0
        observations.removeAll(keepingCapacity: true)
        actions.removeAll(keepingCapacity: true)
        rewards.removeAll(keepingCapacity: true)
        episodeStarts.removeAll(keepingCapacity: true)
        values.removeAll(keepingCapacity: true)
        logProbs.removeAll(keepingCapacity: true)
        advantages.removeAll(keepingCapacity: true)
        returns.removeAll(keepingCapacity: true)
    }

    public mutating func append(_ step: RolloutStep) {
        precondition(count < bufferSize, "RolloutBuffer is full")
        let action = Self.prepareAction(step.action, actionSpace: actionSpace, actionDim: actionDim)
        let reward = Self.scalarFloat(step.reward)
        let episodeStart = Self.scalarFloat(step.episodeStart)
        let value = Self.scalarFloat(step.value)
        let logProb = Self.scalarFloat(step.logProb)

        eval(step.observation, action, step.reward, step.episodeStart, step.value, step.logProb)

        observations.append(step.observation)
        actions.append(action)
        rewards.append(reward)
        episodeStarts.append(episodeStart)
        values.append(value)
        logProbs.append(logProb)
        advantages.append(0.0)
        returns.append(0.0)
        count += 1
    }

    public mutating func computeReturnsAndAdvantages(
        lastValues: MLXArray,
        dones: MLXArray,
        gamma: Double,
        gaeLambda: Double
    ) {
        guard count > 0 else { return }
        let gammaF = Float(gamma)
        let gaeLambdaF = Float(gaeLambda)
        let doneValue = Self.scalarFloat(dones)
        let lastValue = Self.scalarFloat(lastValues)

        var lastGAE: Float = 0.0
        for step in stride(from: count - 1, through: 0, by: -1) {
            let nextNonTerminal: Float
            let nextValue: Float
            if step == count - 1 {
                nextNonTerminal = 1.0 - doneValue
                nextValue = lastValue
            } else {
                nextNonTerminal = 1.0 - episodeStarts[step + 1]
                nextValue = values[step + 1]
            }
            let delta = rewards[step] + gammaF * nextValue * nextNonTerminal - values[step]
            lastGAE = delta + gammaF * gaeLambdaF * nextNonTerminal * lastGAE
            advantages[step] = lastGAE
        }

        for index in 0..<count {
            returns[index] = advantages[index] + values[index]
        }
    }

    public func batches(batchSize: Int) -> [RolloutBatch] {
        batches(batchSize: batchSize, key: nil)
    }

    public func batches(batchSize: Int, key: MLXArray?) -> [RolloutBatch] {
        guard count > 0 else { return [] }
        let safeBatchSize = max(1, min(batchSize, count))
        let permutation: [Int]
        if let key {
            let randomValues = MLX.uniform(0.0..<1.0, [count], key: key, stream: .cpu)
            eval(randomValues)
            let randomList = randomValues.asArray(Float.self)
            permutation = (0..<count).sorted { randomList[$0] < randomList[$1] }
        } else {
            permutation = Array(0..<count).shuffled()
        }
        var result: [RolloutBatch] = []
        result.reserveCapacity((count + safeBatchSize - 1) / safeBatchSize)

        var start = 0
        while start < count {
            let end = min(start + safeBatchSize, count)
            let indices = Array(permutation[start..<end])
            let obsBatch = MLX.stacked(indices.map { observations[$0] }, axis: 0)
            let actionBatch = MLX.stacked(indices.map { actions[$0] }, axis: 0)
            let valuesBatch = MLXArray(indices.map { values[$0] })
            let logProbsBatch = MLXArray(indices.map { logProbs[$0] })
            let advantagesBatch = MLXArray(indices.map { advantages[$0] })
            let returnsBatch = MLXArray(indices.map { returns[$0] })
            eval(obsBatch, actionBatch, valuesBatch, logProbsBatch, advantagesBatch, returnsBatch)
            result.append(
                RolloutBatch(
                    observations: obsBatch,
                    actions: actionBatch,
                    values: valuesBatch,
                    logProbs: logProbsBatch,
                    advantages: advantagesBatch,
                    returns: returnsBatch
                )
            )
            start = end
        }

        return result
    }

    public func valuesAndReturns() -> (values: [Float], returns: [Float]) {
        (
            values: Array(values.prefix(count)),
            returns: Array(returns.prefix(count))
        )
    }

    private static func scalarFloat(_ value: MLXArray) -> Float {
        let casted = value.asType(.float32)
        let flattened = casted.reshaped([-1])
        eval(flattened)
        return flattened.item(Float.self)
    }

    private static func actionDimension(_ actionSpace: any Space) -> Int {
        if let box = boxSpace(from: actionSpace) {
            return box.shape?.reduce(1, *) ?? 1
        }
        if actionSpace is Discrete {
            return 1
        }
        if let multiDiscrete = actionSpace as? MultiDiscrete {
            return multiDiscrete.shape?.reduce(1, *) ?? 1
        }
        if let multiBinary = actionSpace as? MultiBinary {
            return multiBinary.shape?.reduce(1, *) ?? 1
        }
        return actionSpace.shape?.reduce(1, *) ?? 1
    }

    private static func prepareAction(
        _ action: MLXArray,
        actionSpace: any Space,
        actionDim: Int
    ) -> MLXArray {
        if boxSpace(from: actionSpace) != nil {
            return action.asType(.float32).reshaped([actionDim])
        }
        if actionSpace is Discrete {
            var discreteAction = action.asType(.int32)
            if discreteAction.ndim == 0 {
                discreteAction = discreteAction.reshaped([1])
            } else if discreteAction.ndim > 1 {
                discreteAction = discreteAction.reshaped([-1])
                if discreteAction.shape[0] != 1 {
                    discreteAction = discreteAction[0].reshaped([1])
                }
            }
            return discreteAction
        }
        if actionSpace is MultiDiscrete {
            return action.asType(.int32).reshaped([actionDim])
        }
        if actionSpace is MultiBinary {
            return action.asType(.float32).reshaped([actionDim])
        }
        return action.reshaped([actionDim])
    }
}
