import MLX

/// Converts a typed action into replay-buffer elements.
public protocol ReplayActionValue: Sendable {
    /// Returns the flattened action values stored in replay.
    ///
    /// - Returns: Scalar action elements in replay order.
    var replayElements: [Float] { get }
}

/// Converts a typed scalar into a replay-buffer value.
public protocol ReplayScalarValue: Sendable {
    /// Returns the scalar value stored in replay.
    ///
    /// - Returns: The scalar converted to a replay float.
    var replayValue: Float { get }
}

extension Int: ReplayActionValue {
    /// Returns the integer action as one replay element.
    public var replayElements: [Float] {
        [Float(self)]
    }
}

extension Float: ReplayScalarValue {
    /// Returns the floating-point scalar unchanged.
    public var replayValue: Float {
        self
    }
}

extension Int: ReplayScalarValue {
    /// Converts the integer scalar to a replay float.
    public var replayValue: Float {
        Float(self)
    }
}

extension Bool: ReplayScalarValue {
    /// Converts `true` and `false` to one and zero.
    public var replayValue: Float {
        self ? 1 : 0
    }
}

extension TensorValue: ReplayActionValue where Element: ReplayScalarValue {
    /// Flattens scalar tensor elements for replay storage.
    public var replayElements: [Float] {
        elements.map(\.replayValue)
    }
}

extension ReplayBuffer {
    /// Adds typed collection frames to the replay buffer in one batch.
    ///
    /// - Parameter collection: Typed frames to convert and insert.
    /// - Throws: If collection size, frame dimensions, or action dimensions are inconsistent.
    public mutating func add<Action: ReplayActionValue>(
        _ collection: ValueCollection<TensorValue<Float>, Action>
    ) throws {
        guard collection.reset.observations.count == numEnvs else {
            throw GymnazoError.invalidConfiguration(
                "Collection environment count must match replay environment count."
            )
        }
        guard numEnvs == 1 || (!config.optimizeMemoryUsage && config.frameStack == nil) else {
            throw GymnazoError.invalidConfiguration(
                "Multi-environment replay requires independent next observations and uncompressed frames."
            )
        }
        var observationValues: [TensorValue<Float>] = []
        var nextObservationValues: [TensorValue<Float>] = []
        var actionValues: [[Float]] = []
        var rewardValues: [Float] = []
        var terminationValues: [Bool] = []
        var truncationValues: [Bool] = []

        for frame in collection.frames {
            guard frame.observations.count == numEnvs,
                frame.observations.count == frame.actions.count,
                frame.observations.count == frame.step.observations.count,
                frame.observations.count == frame.step.rewards.count,
                frame.observations.count == frame.step.terminations.count,
                frame.observations.count == frame.step.truncations.count,
                frame.observations.count == frame.step.resetInfos.count,
                frame.observations.count == frame.step.finalObservations.count
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

        try addBatch(
            obs: observationBatch,
            actions: actionBatch,
            rewards: rewardBatch,
            nextObs: nextObservationBatch,
            terminations: terminationValues,
            truncations: truncationValues
        )
    }
}
