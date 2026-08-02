import Foundation

/// Typed Swift-state Acrobot environment.
public struct AcrobotValueEnvironment: ValueEnvironment {
    public let dt: Float = 0.2
    public let linkLength1: Float = 1
    public let linkMass1: Float = 1
    public let linkMass2: Float = 1
    public let linkCOMPos1: Float = 0.5
    public let linkCOMPos2: Float = 0.5
    public let linkMOI: Float = 1
    public let gravity: Float = 9.8
    public let maxVel1: Float = 4 * .pi
    public let maxVel2: Float = 9 * .pi
    public let availableTorques: [Float] = [-1, 0, 1]
    public let torqueNoiseMax: Float
    public let useNipsDynamics: Bool

    public let valueObservationSpace: BoxValueSpace
    public let valueActionSpace: DiscreteValueSpace
    public private(set) var state: [Float]?

    private var generator: SeededGenerator?
    private var needsReset = true
    private var isClosed = false

    /// Creates an Acrobot environment with optional torque noise and dynamics.
    ///
    /// - Parameters:
    ///   - torqueNoiseMax: Maximum magnitude of torque noise.
    ///   - useNipsDynamics: Whether to use the alternate dynamics equation.
    /// - Throws: If torqueNoiseMax is negative or non-finite.
    public init(torqueNoiseMax: Float = 0, useNipsDynamics: Bool = false) throws {
        guard torqueNoiseMax.isFinite, torqueNoiseMax >= 0 else {
            throw GymnazoError.invalidConfiguration(
                "Acrobot torque noise must be finite and non-negative."
            )
        }
        self.torqueNoiseMax = torqueNoiseMax
        self.useNipsDynamics = useNipsDynamics
        self.valueActionSpace = try DiscreteValueSpace(n: 3)
        self.valueObservationSpace = try BoxValueSpace(
            low: try TensorValue(
                [-1, -1, -1, -1, -maxVel1, -maxVel2],
                shape: [6]
            ),
            high: try TensorValue(
                [1, 1, 1, 1, maxVel1, maxVel2],
                shape: [6]
            )
        )
    }

    /// Resets both links to a seeded angular state.
    ///
    /// - Parameters:
    ///   - seed: Optional deterministic reset seed.
    ///   - options: Optional low and high initial-state bounds.
    /// - Returns: The initial observation and reset metadata.
    /// - Throws: If options or the environment state are invalid.
    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<TensorValue<Float>> {
        try ensureOpen()
        let options = options ?? [:]
        let supported = Set(["low", "high"])
        let unsupported = Set(options.keys).subtracting(supported)
        guard unsupported.isEmpty else {
            throw GymnazoError.invalidConfiguration(
                "Acrobot does not support reset options: \(unsupported.sorted().joined(separator: ", "))."
            )
        }
        let low = try scalar(options["low"], key: "low", default: -0.1)
        let high = try scalar(options["high"], key: "high", default: 0.1)
        guard low <= high else {
            throw GymnazoError.invalidConfiguration("Acrobot reset bounds must be ordered.")
        }
        if let seed {
            generator = SeededGenerator(seed: seed)
        } else if generator == nil {
            generator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
        }
        guard var generator else {
            throw GymnazoError.invalidState("Acrobot random state is unavailable.")
        }
        state = (0..<4).map { _ in generator.uniform(in: low...high) }
        self.generator = generator
        needsReset = false
        return ValueReset(observation: try observation())
    }

    /// Advances Acrobot with one of its three discrete torques.
    ///
    /// - Parameter action: Discrete torque index from zero through two.
    /// - Returns: The next observation and step result.
    /// - Throws: If the environment is closed, not reset, or action is invalid.
    public mutating func stepValue(_ action: Int) throws -> ValueStep<TensorValue<Float>> {
        try ensureOpen()
        guard !needsReset, let state else {
            throw GymnazoError.stepBeforeReset
        }
        guard valueActionSpace.containsValue(action) else {
            throw GymnazoError.invalidAction("Acrobot actions must be 0, 1, or 2.")
        }
        var torque = availableTorques[action]
        if torqueNoiseMax > 0 {
            guard var generator else {
                throw GymnazoError.invalidState("Acrobot random state is unavailable.")
            }
            torque += generator.uniform(in: -torqueNoiseMax...torqueNoiseMax)
            self.generator = generator
        }
        var next = integrate(state + [torque])
        next[0] = wrap(next[0], min: -.pi, max: .pi)
        next[1] = wrap(next[1], min: -.pi, max: .pi)
        next[2] = min(max(next[2], -maxVel1), maxVel1)
        next[3] = min(max(next[3], -maxVel2), maxVel2)
        self.state = Array(next[0..<4])
        let terminated = isTerminal(next)
        needsReset = terminated
        return ValueStep(
            observation: try observation(),
            reward: terminated ? 0 : -1,
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
            throw GymnazoError.invalidState("Acrobot state is unavailable.")
        }
        return try TensorValue(
            [
                cos(state[0]),
                sin(state[0]),
                cos(state[1]),
                sin(state[1]),
                state[2],
                state[3],
            ],
            shape: [6]
        )
    }

    private func integrate(_ state: [Float]) -> [Float] {
        let halfStep = dt / 2
        let k1 = derivative(state)
        let k2 = derivative(zip(state, k1).map { $0 + halfStep * $1 })
        let k3 = derivative(zip(state, k2).map { $0 + halfStep * $1 })
        let k4 = derivative(zip(state, k3).map { $0 + dt * $1 })
        return state.indices.map { index in
            state[index]
                + dt / 6 * (k1[index] + 2 * k2[index] + 2 * k3[index] + k4[index])
        }
    }

    private func derivative(_ state: [Float]) -> [Float] {
        let theta1 = state[0]
        let theta2 = state[1]
        let velocity1 = state[2]
        let velocity2 = state[3]
        let torque = state[4]
        let d1 =
            linkMass1 * linkCOMPos1 * linkCOMPos1
            + linkMass2
            * (linkLength1 * linkLength1 + linkCOMPos2 * linkCOMPos2
                + 2 * linkLength1 * linkCOMPos2 * cos(theta2))
            + 2 * linkMOI
        let d2 =
            linkMass2
            * (linkCOMPos2 * linkCOMPos2
                + linkLength1 * linkCOMPos2 * cos(theta2))
            + linkMOI
        let phi2 =
            linkMass2 * linkCOMPos2 * gravity
            * cos(theta1 + theta2 - .pi / 2)
        let phi1 =
            -linkMass2 * linkLength1 * linkCOMPos2 * velocity2 * velocity2 * sin(theta2)
            - 2 * linkMass2 * linkLength1 * linkCOMPos2 * velocity2 * velocity1 * sin(theta2)
            + (linkMass1 * linkCOMPos1 + linkMass2 * linkLength1)
            * gravity * cos(theta1 - .pi / 2)
            + phi2
        let denominator =
            linkMass2 * linkCOMPos2 * linkCOMPos2 + linkMOI - d2 * d2 / d1
        let acceleration2: Float
        if useNipsDynamics {
            acceleration2 = (torque + d2 / d1 * phi1 - phi2) / denominator
        } else {
            acceleration2 =
                (torque + d2 / d1 * phi1
                    - linkMass2 * linkLength1 * linkCOMPos2
                    * velocity1 * velocity1 * sin(theta2)
                    - phi2) / denominator
        }
        let acceleration1 = -(d2 * acceleration2 + phi1) / d1
        return [velocity1, velocity2, acceleration1, acceleration2, 0]
    }

    private func isTerminal(_ state: [Float]) -> Bool {
        -cos(state[0]) - cos(state[1] + state[0]) > 1
    }

    private func wrap(_ value: Float, min: Float, max: Float) -> Float {
        var result = value
        let width = max - min
        while result > max {
            result -= width
        }
        while result < min {
            result += width
        }
        return result
    }

    private func scalar(
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
        guard result.isFinite else {
            throw GymnazoError.invalidConfiguration("\(key) must be finite.")
        }
        return result
    }

    private func ensureOpen() throws {
        guard !isClosed else {
            throw GymnazoError.invalidState("Acrobot is closed.")
        }
    }
}
