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
    private var rewards: [MLXArray]
    private var episodeStarts: [MLXArray]
    private var values: [MLXArray]
    private var logProbs: [MLXArray]
    private var advantages: [MLXArray]
    private var returns: [MLXArray]

    public init(
        bufferSize: Int,
        observationSpace: any Space,
        actionSpace: any Space,
        numEnvs: Int = 1
    ) {
        precondition(bufferSize > 0, "RolloutBuffer requires bufferSize > 0")
        precondition(numEnvs > 0, "RolloutBuffer requires numEnvs > 0")
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
        let observation = Self.prepareObservation(
            step.observation,
            observationSpace: observationSpace,
            numEnvs: numEnvs
        )
        let action = Self.prepareAction(
            step.action,
            actionSpace: actionSpace,
            actionDim: actionDim,
            numEnvs: numEnvs
        )
        let reward = Self.prepareScalars(step.reward, numEnvs: numEnvs)
        let episodeStart = Self.prepareScalars(step.episodeStart, numEnvs: numEnvs)
        let value = Self.prepareScalars(step.value, numEnvs: numEnvs)
        let logProb = Self.prepareScalars(step.logProb, numEnvs: numEnvs)

        eval(observation, action, reward, episodeStart, value, logProb)

        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        episodeStarts.append(episodeStart)
        values.append(value)
        logProbs.append(logProb)
        advantages.append(MLX.zeros([numEnvs]))
        returns.append(MLX.zeros([numEnvs]))
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
        let doneValue = Self.prepareScalars(dones, numEnvs: numEnvs)
        let lastValue = Self.prepareScalars(lastValues, numEnvs: numEnvs)

        var lastGAE = MLX.zeros([numEnvs])
        for step in stride(from: count - 1, through: 0, by: -1) {
            let nextNonTerminal: MLXArray
            let nextValue: MLXArray
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
        var generator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
        return batches(batchSize: batchSize, generator: &generator)
    }

    func batches(batchSize: Int, generator: inout SeededGenerator) -> [RolloutBatch] {
        guard let rollout = prepared() else { return [] }
        return Array(rollout.batches(batchSize: batchSize, generator: &generator))
    }

    func prepared() -> PreparedRollout? {
        guard count > 0 else { return nil }
        let sampleCount = count * numEnvs
        let batch = RolloutBatch(
            observations: MLX.stacked(observations, axis: 0)
                .reshaped([sampleCount] + (observationSpace.shape ?? [])),
            actions: MLX.stacked(actions, axis: 0).reshaped([sampleCount, actionDim]),
            values: MLX.stacked(values, axis: 0).reshaped([sampleCount]),
            logProbs: MLX.stacked(logProbs, axis: 0).reshaped([sampleCount]),
            advantages: MLX.stacked(advantages, axis: 0).reshaped([sampleCount]),
            returns: MLX.stacked(returns, axis: 0).reshaped([sampleCount])
        )
        eval(
            batch.observations, batch.actions, batch.values,
            batch.logProbs, batch.advantages, batch.returns
        )
        return PreparedRollout(batch: batch)
    }

    public func valuesAndReturns() -> (values: [Float], returns: [Float]) {
        let valueTensor = MLX.stacked(Array(values.prefix(count)), axis: 0).reshaped([-1])
        let returnTensor = MLX.stacked(Array(returns.prefix(count)), axis: 0).reshaped([-1])
        eval(valueTensor, returnTensor)
        return (
            values: valueTensor.asArray(Float.self),
            returns: returnTensor.asArray(Float.self)
        )
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
        actionDim: Int,
        numEnvs: Int
    ) -> MLXArray {
        if boxSpace(from: actionSpace) != nil {
            return action.asType(.float32).reshaped([numEnvs, actionDim])
        }
        if actionSpace is Discrete {
            return action.asType(.int32).reshaped([numEnvs, 1])
        }
        if actionSpace is MultiDiscrete {
            return action.asType(.int32).reshaped([numEnvs, actionDim])
        }
        if actionSpace is MultiBinary {
            return action.asType(.float32).reshaped([numEnvs, actionDim])
        }
        return action.reshaped([numEnvs, actionDim])
    }

    private static func prepareObservation(
        _ observation: MLXArray,
        observationSpace: any Space,
        numEnvs: Int
    ) -> MLXArray {
        let shape = observationSpace.shape ?? []
        return observation.reshaped([numEnvs] + shape)
    }

    private static func prepareScalars(
        _ values: MLXArray,
        numEnvs: Int
    ) -> MLXArray {
        values.asType(.float32).reshaped([numEnvs])
    }
}
