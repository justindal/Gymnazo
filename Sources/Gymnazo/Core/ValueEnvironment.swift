/// Sendable reset options for a typed environment.
public typealias ValueResetOptions = [String: SpaceValue]

/// The typed result returned by a value environment reset.
public struct ValueReset<Observation: Sendable>: Sendable {
    public let observation: Observation
    public let info: ValueInfo

    /// Creates a reset result with optional typed metadata.
    public init(observation: Observation, info: ValueInfo = [:]) {
        self.observation = observation
        self.info = info
    }
}

/// The typed result returned by a value environment step.
public struct ValueStep<Observation: Sendable>: Sendable {
    public let observation: Observation
    public let reward: Double
    public let terminated: Bool
    public let truncated: Bool
    public let info: ValueInfo

    /// Creates a step result with explicit termination and truncation flags.
    public init(
        observation: Observation,
        reward: Double,
        terminated: Bool,
        truncated: Bool,
        info: ValueInfo = [:]
    ) {
        self.observation = observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
    }
}

/// A Swift-value environment whose state can cross concurrency boundaries safely.
public protocol ValueEnvironment: Sendable {
    associatedtype Observation: Sendable
    associatedtype Action: Sendable
    associatedtype ObservationSpace: ValueSpace where ObservationSpace.Value == Observation
    associatedtype ActionSpace: ValueSpace where ActionSpace.Value == Action

    var valueObservationSpace: ObservationSpace { get }
    var valueActionSpace: ActionSpace { get }

    /// Resets state and returns the initial typed observation.
    ///
    /// - Parameters:
    ///   - seed: Optional deterministic reset seed.
    ///   - options: Environment-specific reset options.
    /// - Returns: The initial observation and reset metadata.
    /// - Throws: If the reset options or environment state are invalid.
    mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation>
    /// Applies one typed action and returns the next typed step.
    ///
    /// - Parameter action: The typed action to apply.
    /// - Returns: The next observation, reward, terminal flags, and metadata.
    /// - Throws: If the action or environment state is invalid.
    mutating func stepValue(_ action: Action) throws -> ValueStep<Observation>
    /// Releases environment resources.
    mutating func close()
}

extension ValueEnvironment {
    /// Resets without supplying typed options.
    ///
    /// - Parameter seed: Optional deterministic reset seed.
    /// - Returns: The initial observation and reset metadata.
    /// - Throws: If the reset fails.
    public mutating func resetValue(seed: UInt64? = nil) throws -> ValueReset<Observation> {
        try resetValue(seed: seed, options: nil)
    }
}

/// Adds lifecycle and space-contract checks to a value environment.
public struct CheckedValueEnvironment<Base: ValueEnvironment>: ValueEnvironment {
    public typealias Observation = Base.Observation
    public typealias Action = Base.Action
    public typealias ObservationSpace = Base.ObservationSpace
    public typealias ActionSpace = Base.ActionSpace

    public var base: Base
    private var state: State = .initial

    public var valueObservationSpace: ObservationSpace {
        base.valueObservationSpace
    }

    public var valueActionSpace: ActionSpace {
        base.valueActionSpace
    }

    /// Wraps a value environment with contract validation.
    ///
    /// - Parameter base: The environment whose lifecycle and spaces should be checked.
    public init(_ base: Base) {
        self.base = base
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation> {
        try ensureOpen()
        let result = try base.resetValue(seed: seed, options: options)
        guard valueObservationSpace.containsValue(result.observation) else {
            throw GymnazoError.observationOutsideSpace(envId: nil)
        }
        state = .active
        return result
    }

    public mutating func stepValue(_ action: Action) throws -> ValueStep<Observation> {
        switch state {
        case .initial:
            throw GymnazoError.stepBeforeReset
        case .finished:
            throw GymnazoError.invalidState(
                "Reset the environment before stepping after an episode ends.")
        case .closed:
            throw GymnazoError.invalidState("The environment is closed.")
        case .active:
            break
        }

        guard valueActionSpace.containsValue(action) else {
            throw GymnazoError.actionOutsideSpace(envId: nil)
        }

        let result = try base.stepValue(action)
        guard valueObservationSpace.containsValue(result.observation) else {
            throw GymnazoError.observationOutsideSpace(envId: nil)
        }
        if result.terminated || result.truncated {
            state = .finished
        }
        return result
    }

    public mutating func close() {
        guard state != .closed else { return }
        base.close()
        state = .closed
    }

    private func ensureOpen() throws {
        guard state != .closed else {
            throw GymnazoError.invalidState("The environment is closed.")
        }
    }

    private enum State {
        case initial
        case active
        case finished
        case closed
    }
}
