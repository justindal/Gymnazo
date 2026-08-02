import MLX

public protocol ReplayActionValue: Sendable {
    var replayElements: [Float] { get }
}

public protocol ReplayScalarValue: Sendable {
    var replayValue: Float { get }
}

extension Int: ReplayActionValue {
    public var replayElements: [Float] {
        [Float(self)]
    }
}

extension Float: ReplayScalarValue {
    public var replayValue: Float {
        self
    }
}

extension Int: ReplayScalarValue {
    public var replayValue: Float {
        Float(self)
    }
}

extension Bool: ReplayScalarValue {
    public var replayValue: Float {
        self ? 1 : 0
    }
}

extension TensorValue: ReplayActionValue where Element: ReplayScalarValue {
    public var replayElements: [Float] {
        elements.map(\.replayValue)
    }
}

extension ReplayBuffer {
    public mutating func add<Action: ReplayActionValue>(
        _ collection: ValueCollection<TensorValue<Float>, Action>
    ) throws {
        var observationValues: [TensorValue<Float>] = []
        var nextObservationValues: [TensorValue<Float>] = []
        var actionValues: [[Float]] = []
        var rewardValues: [Float] = []
        var terminationValues: [Bool] = []
        var truncationValues: [Bool] = []

        for frame in collection.frames {
            guard frame.observations.count == frame.actions.count,
                frame.observations.count == frame.step.observations.count
            else {
                throw GymnazoError.invalidState(
                    "A value collection frame has inconsistent batch dimensions."
                )
            }
            for index in frame.observations.indices {
                if frame.step.resetInfos[index] != nil
                    && frame.step.finalObservations[index] == nil
                {
                    continue
                }
                observationValues.append(frame.observations[index])
                actionValues.append(frame.actions[index].replayElements)
                rewardValues.append(Float(frame.step.rewards[index]))
                terminationValues.append(frame.step.terminations[index])
                truncationValues.append(frame.step.truncations[index])
                nextObservationValues.append(
                    frame.step.finalObservations[index] ?? frame.step.observations[index]
                )
            }
        }
        guard !observationValues.isEmpty else { return }
        guard let actionDimension = actionValues.first?.count,
            actionValues.allSatisfy({ $0.count == actionDimension })
        else {
            throw GymnazoError.invalidState("Replay actions have inconsistent dimensions.")
        }

        let observationBatch = try MLXValueCodec.encodeBatch(observationValues)
        let nextObservationBatch = try MLXValueCodec.encodeBatch(nextObservationValues)
        let actionBatch = MLXArray(
            actionValues.flatMap { $0 },
            [actionValues.count, actionDimension]
        )
        let rewardBatch = MLXArray(rewardValues)

        for index in observationValues.indices {
            try add(
                obs: observationBatch[index],
                action: actionBatch[index],
                reward: rewardBatch[index],
                nextObs: nextObservationBatch[index],
                terminated: terminationValues[index],
                truncated: truncationValues[index]
            )
        }
        if let nextObservations {
            eval(observations, nextObservations, actions, rewards, dones, timeouts)
        } else {
            eval(observations, actions, rewards, dones, timeouts)
        }
    }
}
