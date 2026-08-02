import Foundation

/// Typed Swift-state continuous-control MountainCar environment.
public struct MountainCarContinuousValueEnvironment: ValueEnvironment {
    public let minPosition: Float = -1.2
    public let maxPosition: Float = 0.6
    public let maxSpeed: Float = 0.07
    public let goalPosition: Float = 0.45
    public let goalVelocity: Float
    public let power: Float = 0.0015
    public let gravity: Float = 0.0025

    public let valueObservationSpace: BoxValueSpace
    public let valueActionSpace: BoxValueSpace
    public private(set) var state: (position: Float, velocity: Float)?

    private var generator: SeededGenerator?
    private var needsReset = true
    private var isClosed = false

    /// Creates a continuous MountainCar environment with the requested goal velocity.
    ///
    /// - Parameter goalVelocity: Minimum velocity required at the goal.
    /// - Throws: If the configuration is invalid.
    public init(goalVelocity: Float = 0) throws {
        guard goalVelocity.isFinite else {
            throw GymnazoError.invalidConfiguration("Goal velocity must be finite.")
        }
        self.goalVelocity = goalVelocity
        self.valueActionSpace = try BoxValueSpace(low: -1, high: 1, shape: [1])
        self.valueObservationSpace = try BoxValueSpace(
            low: try TensorValue([minPosition, -maxSpeed], shape: [2]),
            high: try TensorValue([maxPosition, maxSpeed], shape: [2])
        )
    }

    /// Resets the car to a seeded position in the starting interval.
    ///
    /// - Parameters:
    ///   - seed: Optional deterministic reset seed.
    ///   - options: MountainCarContinuous does not support reset options.
    /// - Returns: The initial observation and reset metadata.
    /// - Throws: If the environment is closed or options are supplied.
    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<TensorValue<Float>> {
        try ensureOpen()
        guard options?.isEmpty != false else {
            throw GymnazoError.invalidConfiguration(
                "MountainCarContinuous does not support reset options."
            )
        }
        if let seed {
            generator = SeededGenerator(seed: seed)
        } else if generator == nil {
            generator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
        }
        guard var generator else {
            throw GymnazoError.invalidState("MountainCarContinuous random state is unavailable.")
        }
        state = (generator.uniform(in: -0.6 ... -0.4), 0)
        self.generator = generator
        needsReset = false
        return ValueReset(observation: try observation())
    }

    /// Advances the car with one bounded continuous force.
    ///
    /// - Parameter action: A one-element tensor containing the force.
    /// - Returns: The next observation and step result.
    /// - Throws: If the environment is closed, not reset, or action is invalid.
    public mutating func stepValue(
        _ action: TensorValue<Float>
    ) throws -> ValueStep<TensorValue<Float>> {
        try ensureOpen()
        guard !needsReset, var state else {
            throw GymnazoError.stepBeforeReset
        }
        guard action.shape == [1], let requestedForce = action.elements.first,
            requestedForce.isFinite
        else {
            throw GymnazoError.invalidAction(
                "MountainCarContinuous requires one finite force value."
            )
        }
        let force = min(max(requestedForce, -1), 1)
        state.velocity += force * power - cos(3 * state.position) * gravity
        state.velocity = min(max(state.velocity, -maxSpeed), maxSpeed)
        state.position = min(max(state.position + state.velocity, minPosition), maxPosition)
        if state.position == minPosition, state.velocity < 0 {
            state.velocity = 0
        }
        self.state = state
        let terminated = state.position >= goalPosition && state.velocity >= goalVelocity
        needsReset = terminated
        return ValueStep(
            observation: try observation(),
            reward: (terminated ? 100 : 0) - Double(force * force) * 0.1,
            terminated: terminated,
            truncated: false
        )
    }

    public mutating func close() {
        isClosed = true
        needsReset = true
        state = nil
    }

    private func observation() throws -> TensorValue<Float> {
        guard let state else {
            throw GymnazoError.invalidState("MountainCarContinuous state is unavailable.")
        }
        return try TensorValue([state.position, state.velocity], shape: [2])
    }

    private func ensureOpen() throws {
        guard !isClosed else {
            throw GymnazoError.invalidState("MountainCarContinuous is closed.")
        }
    }
}
