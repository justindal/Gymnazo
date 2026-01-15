//
// Wrapper.swift
//

import MLX

/// protocol for environment wrappers.
/// is generic over the `BaseEnv` it wraps for better type safety.
public protocol Wrapper<BaseEnv>: Env
where
    Observation == BaseEnv.Observation,
    Action == BaseEnv.Action,
    ObservationSpace == BaseEnv.ObservationSpace,
    ActionSpace == BaseEnv.ActionSpace
{
    associatedtype BaseEnv: Env

    /// "inner" environment instance.
    var env: BaseEnv { get set }

    /// requires that all wrappers can be initialized with the
    /// environment they are wrapping.
    init(env: BaseEnv)
}

/// This extension provides the "pass-through" logic. By default,
/// a Wrapper just forwards all calls and properties to its
/// inner `env`.
extension Wrapper {
    public var actionSpace: BaseEnv.ActionSpace {
        return env.actionSpace
    }

    public var observationSpace: BaseEnv.ObservationSpace {
        return env.observationSpace
    }

    public var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }

    public var renderMode: String? {
        get { env.renderMode }
        set { env.renderMode = newValue }
    }

    public var unwrapped: any Env {
        return env.unwrapped
    }

    public mutating func step(_ action: Action) -> Step<Observation> {
        return env.step(action)
    }

    public mutating func reset(
        seed: UInt64?,
        options: [String: Any]?
    ) -> Reset<Observation> {
        return env.reset(seed: seed, options: options)
    }

    public func close() {
        return env.close()
    }

    @discardableResult
    public func render() -> Any? {
        return env.render()
    }
}
