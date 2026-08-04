public struct TwentyFortyEightValueEnvironment: ValueEnvironment {
    public let valueObservationSpace: BoxValueSpace
    public let valueActionSpace: DiscreteValueSpace

    private var model = TwentyFortyEightModel()
    private var isClosed = false

    public init() throws {
        self.valueObservationSpace = try BoxValueSpace(
            low: 0,
            high: 1,
            shape: [TwentyFortyEightModel.observationCount]
        )
        self.valueActionSpace = try DiscreteValueSpace(n: 4)
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<TensorValue<Float>> {
        try ensureOpen()
        guard options?.isEmpty != false else {
            throw GymnazoError.invalidConfiguration(
                "2048 does not support reset options."
            )
        }
        let values = model.reset(seed: seed)
        let snapshot = model.snapshot
        let observation = try TensorValue(
            values,
            shape: [TwentyFortyEightModel.observationCount]
        )
        return ValueReset(
            observation: observation,
            info: [
                "score": .discrete(snapshot.score),
                "max_tile": .discrete(snapshot.maxTile),
                "game_over": .bool(snapshot.isGameOver),
            ]
        )
    }

    public mutating func stepValue(_ action: Int) throws -> ValueStep<TensorValue<Float>> {
        try ensureOpen()
        guard valueActionSpace.containsValue(action) else {
            throw GymnazoError.invalidAction("2048 actions must be 0, 1, 2, or 3.")
        }

        let result = try model.step(action: action)
        return ValueStep(
            observation: try TensorValue(
                result.observation,
                shape: [TwentyFortyEightModel.observationCount]
            ),
            reward: result.reward,
            terminated: result.terminated,
            truncated: false,
            info: [
                "score": .discrete(result.score),
                "max_tile": .discrete(result.maxTile),
                "game_over": .bool(result.terminated),
            ]
        )
    }

    public mutating func close() {
        isClosed = true
    }

    private func ensureOpen() throws {
        guard !isClosed else {
            throw GymnazoError.invalidState("2048 is closed.")
        }
    }
}
