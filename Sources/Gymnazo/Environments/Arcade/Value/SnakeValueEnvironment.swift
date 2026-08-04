/// Typed Swift-state Snake environment.
public struct SnakeValueEnvironment: ValueEnvironment {
    public let valueObservationSpace: BoxValueSpace
    public let valueActionSpace: DiscreteValueSpace

    private var model = SnakeModel()
    private var isClosed = false

    /// Creates the standard typed Snake environment.
    public init() throws {
        self.valueObservationSpace = try BoxValueSpace(
            low: 0,
            high: 1,
            shape: [SnakeModel.observationCount]
        )
        self.valueActionSpace = try DiscreteValueSpace(n: 3)
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<TensorValue<Float>> {
        try ensureOpen()
        guard options?.isEmpty != false else {
            throw GymnazoError.invalidConfiguration("Snake does not support reset options.")
        }
        let observation = try TensorValue(
            model.reset(seed: seed),
            shape: [SnakeModel.observationCount]
        )
        return ValueReset(
            observation: observation,
            info: ["score": .discrete(0), "won": .bool(false)]
        )
    }

    public mutating func stepValue(_ action: Int) throws -> ValueStep<TensorValue<Float>> {
        try ensureOpen()
        guard valueActionSpace.containsValue(action) else {
            throw GymnazoError.invalidAction("Snake actions must be 0, 1, or 2.")
        }

        let result = try model.step(action: action)
        return ValueStep(
            observation: try TensorValue(
                result.observation,
                shape: [SnakeModel.observationCount]
            ),
            reward: result.reward,
            terminated: result.terminated,
            truncated: false,
            info: [
                "score": .discrete(result.score),
                "won": .bool(result.won),
            ]
        )
    }

    public mutating func close() {
        isClosed = true
    }

    private func ensureOpen() throws {
        guard !isClosed else {
            throw GymnazoError.invalidState("Snake is closed.")
        }
    }
}
