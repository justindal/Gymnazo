import Foundation

public struct MountainCarValueEnvironment: ValueEnvironment {
    public let minPosition: Float = -1.2
    public let maxPosition: Float = 0.6
    public let maxSpeed: Float = 0.07
    public let goalPosition: Float = 0.5
    public let goalVelocity: Float
    public let force: Float = 0.001
    public let gravity: Float = 0.0025

    public let valueObservationSpace: BoxValueSpace
    public let valueActionSpace: DiscreteValueSpace

    public private(set) var state: [Float]?

    private var generator: SeededGenerator?
    private var needsReset = true
    private var isClosed = false

    public init(goalVelocity: Float = 0) throws {
        self.goalVelocity = goalVelocity
        self.valueActionSpace = try DiscreteValueSpace(n: 3)
        self.valueObservationSpace = try BoxValueSpace(
            low: try TensorValue([minPosition, -maxSpeed], shape: [2]),
            high: try TensorValue([maxPosition, maxSpeed], shape: [2])
        )
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<TensorValue<Float>> {
        try ensureOpen()
        guard options?.isEmpty != false else {
            throw GymnazoError.invalidConfiguration("MountainCar does not support reset options.")
        }
        if let seed {
            generator = SeededGenerator(seed: seed)
        } else if generator == nil {
            generator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
        }
        guard var generator else {
            throw GymnazoError.invalidState("MountainCar random state is unavailable.")
        }
        state = [generator.uniform(in: -0.6 ... -0.4), 0]
        self.generator = generator
        needsReset = false
        return ValueReset(observation: try observation())
    }

    public mutating func stepValue(_ action: Int) throws -> ValueStep<TensorValue<Float>> {
        try ensureOpen()
        guard !needsReset, let state else {
            throw GymnazoError.stepBeforeReset
        }
        guard valueActionSpace.containsValue(action) else {
            throw GymnazoError.invalidAction("MountainCar actions must be 0, 1, or 2.")
        }

        var position = state[0]
        var velocity = state[1]
        velocity += Float(action - 1) * force - cos(3 * position) * gravity
        velocity = min(max(velocity, -maxSpeed), maxSpeed)
        position = min(max(position + velocity, minPosition), maxPosition)
        if position == minPosition, velocity < 0 {
            velocity = 0
        }
        self.state = [position, velocity]

        let terminated = position >= goalPosition && velocity >= goalVelocity
        needsReset = terminated
        return ValueStep(
            observation: try observation(),
            reward: -1,
            terminated: terminated,
            truncated: false
        )
    }

    public mutating func close() {
        isClosed = true
        state = nil
        needsReset = true
    }

    private func observation() throws -> TensorValue<Float> {
        guard let state else {
            throw GymnazoError.invalidState("MountainCar state is unavailable.")
        }
        return try TensorValue(state, shape: [2])
    }

    private func ensureOpen() throws {
        guard !isClosed else {
            throw GymnazoError.invalidState("MountainCar is closed.")
        }
    }
}
