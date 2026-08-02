import Foundation

/// Typed Swift-state Pendulum environment.
public struct PendulumValueEnvironment: ValueEnvironment {
    public let maxSpeed: Float = 8
    public let maxTorque: Float = 2
    public let dt: Float = 0.05
    public let g: Float
    public let m: Float = 1
    public let l: Float = 1

    public let valueObservationSpace: BoxValueSpace
    public let valueActionSpace: BoxValueSpace

    public private(set) var state: (theta: Float, thetaDot: Float)?

    private var generator: SeededGenerator?
    private var needsReset = true
    private var isClosed = false

    /// Creates a Pendulum environment with the supplied gravity.
    ///
    /// - Parameter g: Gravitational acceleration.
    /// - Throws: If the configuration is invalid.
    public init(g: Float = 10) throws {
        guard g.isFinite else {
            throw GymnazoError.invalidConfiguration("Pendulum gravity must be finite.")
        }
        self.g = g
        self.valueActionSpace = try BoxValueSpace(
            low: try TensorValue([-maxTorque], shape: [1]),
            high: try TensorValue([maxTorque], shape: [1])
        )
        self.valueObservationSpace = try BoxValueSpace(
            low: try TensorValue([-1, -1, -maxSpeed], shape: [3]),
            high: try TensorValue([1, 1, maxSpeed], shape: [3])
        )
    }

    /// Resets the pendulum to a seeded angle and angular velocity.
    ///
    /// - Parameters:
    ///   - seed: Optional deterministic reset seed.
    ///   - options: Pendulum does not support reset options.
    /// - Returns: The initial observation and reset metadata.
    /// - Throws: If the environment is closed or options are supplied.
    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<TensorValue<Float>> {
        try ensureOpen()
        let options = options ?? [:]
        let supported = Set(["x_init", "y_init"])
        let unsupported = Set(options.keys).subtracting(supported)
        guard unsupported.isEmpty else {
            throw GymnazoError.invalidConfiguration(
                "Pendulum does not support reset options: \(unsupported.sorted().joined(separator: ", "))."
            )
        }
        let xInit = try nonnegativeOption(options["x_init"], key: "x_init", default: .pi)
        let yInit = try nonnegativeOption(options["y_init"], key: "y_init", default: 1)

        if let seed {
            generator = SeededGenerator(seed: seed)
        } else if generator == nil {
            generator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
        }
        guard var generator else {
            throw GymnazoError.invalidState("Pendulum random state is unavailable.")
        }
        state = (
            theta: generator.uniform(in: -xInit...xInit),
            thetaDot: generator.uniform(in: -yInit...yInit)
        )
        self.generator = generator
        needsReset = false
        return ValueReset(observation: try observation())
    }

    /// Advances the pendulum with one bounded torque.
    ///
    /// - Parameter action: A one-element tensor containing the torque.
    /// - Returns: The next observation and step result.
    /// - Throws: If the environment is closed, not reset, or action is invalid.
    public mutating func stepValue(
        _ action: TensorValue<Float>
    ) throws -> ValueStep<TensorValue<Float>> {
        try ensureOpen()
        guard !needsReset, let state else {
            throw GymnazoError.stepBeforeReset
        }
        guard action.shape == [1], let requestedTorque = action.elements.first,
            requestedTorque.isFinite
        else {
            throw GymnazoError.invalidAction("Pendulum requires one finite torque value.")
        }

        let torque = min(max(requestedTorque, -maxTorque), maxTorque)
        let normalizedTheta = angleNormalize(state.theta)
        let cost =
            normalizedTheta * normalizedTheta
            + 0.1 * state.thetaDot * state.thetaDot
            + 0.001 * torque * torque
        let velocity =
            state.thetaDot
            + (3 * g / (2 * l) * sin(state.theta) + 3 / (m * l * l) * torque) * dt
        let clippedVelocity = min(max(velocity, -maxSpeed), maxSpeed)
        self.state = (
            theta: state.theta + clippedVelocity * dt,
            thetaDot: clippedVelocity
        )
        return ValueStep(
            observation: try observation(),
            reward: Double(-cost),
            terminated: false,
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
            throw GymnazoError.invalidState("Pendulum state is unavailable.")
        }
        return try TensorValue(
            [cos(state.theta), sin(state.theta), state.thetaDot],
            shape: [3]
        )
    }

    private func nonnegativeOption(
        _ value: SpaceValue?,
        key: String,
        default defaultValue: Float
    ) throws -> Float {
        let result: Float
        switch value {
        case nil:
            result = defaultValue
        case .double(let value):
            result = Float(value)
        case .continuous(let values, let shape) where shape == [1] && values.count == 1:
            result = values[0]
        case .floatTensor(let value) where value.shape == [1] && value.count == 1:
            result = value.elements[0]
        default:
            throw GymnazoError.invalidConfiguration("\(key) must be a scalar.")
        }
        guard result.isFinite, result >= 0 else {
            throw GymnazoError.invalidConfiguration("\(key) must be finite and non-negative.")
        }
        return result
    }

    private func angleNormalize(_ value: Float) -> Float {
        let twoPi = 2 * Float.pi
        return ((value + .pi).truncatingRemainder(dividingBy: twoPi) + twoPi)
            .truncatingRemainder(dividingBy: twoPi) - .pi
    }

    private func ensureOpen() throws {
        guard !isClosed else {
            throw GymnazoError.invalidState("Pendulum is closed.")
        }
    }
}
