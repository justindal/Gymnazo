public typealias ValueResetOptions = [String: SpaceValue]

public struct ValueReset<Observation: Sendable>: Sendable {
    public let observation: Observation
    public let info: ValueInfo

    public init(observation: Observation, info: ValueInfo = [:]) {
        self.observation = observation
        self.info = info
    }
}

public struct ValueStep<Observation: Sendable>: Sendable {
    public let observation: Observation
    public let reward: Double
    public let terminated: Bool
    public let truncated: Bool
    public let info: ValueInfo

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

public protocol ValueEnvironment: Sendable {
    associatedtype Observation: Sendable
    associatedtype Action: Sendable
    associatedtype ObservationSpace: ValueSpace where ObservationSpace.Value == Observation
    associatedtype ActionSpace: ValueSpace where ActionSpace.Value == Action

    var valueObservationSpace: ObservationSpace { get }
    var valueActionSpace: ActionSpace { get }

    mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation>
    mutating func stepValue(_ action: Action) throws -> ValueStep<Observation>
    mutating func close()
}

extension ValueEnvironment {
    public mutating func resetValue(seed: UInt64? = nil) throws -> ValueReset<Observation> {
        try resetValue(seed: seed, options: nil)
    }
}

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
