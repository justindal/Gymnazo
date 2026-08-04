/// Typed Swift-state Flappy Bird environment.
public struct FlappyBirdValueEnvironment: ValueEnvironment {
    public let valueObservationSpace: BoxValueSpace
    public let valueActionSpace: DiscreteValueSpace

    private var model = FlappyBirdModel()
    private var isClosed = false

    /// Creates the standard typed Flappy Bird environment.
    public init() throws {
        self.valueObservationSpace = try BoxValueSpace(low: -1, high: 1, shape: [4])
        self.valueActionSpace = try DiscreteValueSpace(n: 2)
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<TensorValue<Float>> {
        try ensureOpen()
        guard options?.isEmpty != false else {
            throw GymnazoError.invalidConfiguration(
                "FlappyBird does not support reset options."
            )
        }
        let observation = try TensorValue(model.reset(seed: seed), shape: [4])
        return ValueReset(observation: observation, info: ["score": .discrete(0)])
    }

    public mutating func stepValue(_ action: Int) throws -> ValueStep<TensorValue<Float>> {
        try ensureOpen()
        guard valueActionSpace.containsValue(action) else {
            throw GymnazoError.invalidAction("FlappyBird actions must be 0 or 1.")
        }

        let result = try model.step(flaps: action == 1)
        return ValueStep(
            observation: try TensorValue(result.observation, shape: [4]),
            reward: result.reward,
            terminated: result.terminated,
            truncated: false,
            info: ["score": .discrete(result.score)]
        )
    }

    public mutating func close() {
        isClosed = true
    }

    private func ensureOpen() throws {
        guard !isClosed else {
            throw GymnazoError.invalidState("FlappyBird is closed.")
        }
    }
}
