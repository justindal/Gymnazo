import Foundation

/// Typed Swift-state CartPole environment.
public struct CartPoleValueEnvironment: ValueEnvironment {
    public let gravity: Float = 9.8
    public let cartMass: Float = 1
    public let poleMass: Float = 0.1
    public let poleHalfLength: Float = 0.5
    public let forceMagnitude: Float = 10
    public let timeStep: Float = 0.02
    public let angleThreshold: Float = 12 * 2 * Float.pi / 360
    public let positionThreshold: Float = 2.4

    public let valueObservationSpace: BoxValueSpace
    public let valueActionSpace: DiscreteValueSpace

    public private(set) var state: [Float]?

    private let totalMass: Float
    private let poleMassLength: Float
    private var generator: SeededGenerator?
    private var needsReset = true
    private var isClosed = false

    /// Creates a CartPole environment with the standard dynamics.
    public init() throws {
        self.totalMass = cartMass + poleMass
        self.poleMassLength = poleMass * poleHalfLength
        self.valueActionSpace = try DiscreteValueSpace(n: 2)
        self.valueObservationSpace = try BoxValueSpace(
            low: try TensorValue(
                [
                    -positionThreshold * 2,
                    -Float.greatestFiniteMagnitude,
                    -angleThreshold * 2,
                    -Float.greatestFiniteMagnitude,
                ],
                shape: [4]
            ),
            high: try TensorValue(
                [
                    positionThreshold * 2,
                    Float.greatestFiniteMagnitude,
                    angleThreshold * 2,
                    Float.greatestFiniteMagnitude,
                ],
                shape: [4]
            )
        )
    }

    /// Resets the environment and returns its initial observation.
    ///
    /// - Parameters:
    ///   - seed: Optional deterministic reset seed.
    ///   - options: CartPole does not support reset options.
    /// - Returns: The initial observation and reset metadata.
    /// - Throws: If the environment is closed or options are supplied.
    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<TensorValue<Float>> {
        try ensureOpen()
        guard options?.isEmpty != false else {
            throw GymnazoError.invalidConfiguration("CartPole does not support reset options.")
        }
        if let seed {
            generator = SeededGenerator(seed: seed)
        } else if generator == nil {
            generator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
        }
        guard var generator else {
            throw GymnazoError.invalidState("CartPole random state is unavailable.")
        }
        state = (0..<4).map { _ in generator.uniform(in: -0.05...0.05) }
        self.generator = generator
        needsReset = false
        return ValueReset(observation: try observation())
    }

    /// Advances CartPole using a discrete left or right force.
    ///
    /// - Parameter action: Zero for left force or one for right force.
    /// - Returns: The next observation and step result.
    /// - Throws: If the environment is closed, not reset, or action is invalid.
    public mutating func stepValue(_ action: Int) throws -> ValueStep<TensorValue<Float>> {
        try ensureOpen()
        guard !needsReset, let state else {
            throw GymnazoError.stepBeforeReset
        }
        guard valueActionSpace.containsValue(action) else {
            throw GymnazoError.invalidAction("CartPole actions must be 0 or 1.")
        }

        let position = state[0]
        let velocity = state[1]
        let angle = state[2]
        let angularVelocity = state[3]
        let force = action == 1 ? forceMagnitude : -forceMagnitude
        let cosine = cos(angle)
        let sine = sin(angle)
        let temporary =
            (force + poleMassLength * angularVelocity * angularVelocity * sine) / totalMass
        let angularAcceleration =
            (gravity * sine - cosine * temporary)
            / (poleHalfLength * (4 / 3 - poleMass * cosine * cosine / totalMass))
        let acceleration = temporary - poleMassLength * angularAcceleration * cosine / totalMass

        let nextState = [
            position + timeStep * velocity,
            velocity + timeStep * acceleration,
            angle + timeStep * angularVelocity,
            angularVelocity + timeStep * angularAcceleration,
        ]
        self.state = nextState

        let terminated =
            nextState[0] < -positionThreshold
            || nextState[0] > positionThreshold
            || nextState[2] < -angleThreshold
            || nextState[2] > angleThreshold
        needsReset = terminated

        return ValueStep(
            observation: try observation(),
            reward: 1,
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
            throw GymnazoError.invalidState("CartPole state is unavailable.")
        }
        return try TensorValue(state, shape: [4])
    }

    private func ensureOpen() throws {
        guard !isClosed else {
            throw GymnazoError.invalidState("CartPole is closed.")
        }
    }
}
