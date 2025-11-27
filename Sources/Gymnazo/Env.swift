//
//  Env.swift
//

public protocol Env<Observation, Action> {
    associatedtype Observation
    associatedtype Action

    associatedtype ObservationSpace: Space where ObservationSpace.T == Observation
    associatedtype ActionSpace: Space where ActionSpace.T == Action

    typealias StepResult = (
        obs: Observation,
        reward: Double,
        terminated: Bool,
        truncated: Bool,
        info: [String: Any]
    )

    typealias ResetResult = (
        obs: Observation,
        info: [String: Any]
    )

    var action_space: ActionSpace { get }
    var observation_space: ObservationSpace { get }
    var spec: EnvSpec? { get set }
    var render_mode: String? { get set }

    var unwrapped: any Env { get }
    
    mutating func step(_ action: Action) -> StepResult

    /// resets the environment to an initial state, returning an initial observation and info.
    /// this generates a new starting state, often with some randomness controlled by the optional seed parameter.
    mutating func reset(
        seed: UInt64?,
        options: [String: Any]?
    ) -> ResetResult
    
    /// renders the environment.
    /// the return value depends on the `render_mode`.
    /// - "ansi": returns a `String`
    /// - "rgb_array": returns a `CGImage` (or platform equivalent)
    /// - "human": returns `nil`
    @discardableResult
    func render() -> Any?
}

public extension Env {
    /// override for `reset`
    mutating func reset(seed: UInt64) -> ResetResult {
        return self.reset(seed: seed, options: nil)
    }
    
    /// override for `reset` with no seed
    mutating func reset() -> ResetResult {
        return self.reset(seed: nil, options: nil)
    }

    var unwrapped: any Env {
        self
    }

    @discardableResult
    func render() -> Any? {
        return nil
    }
    
    func close() {
        // do nothing
    }
}