//
// Wrapper.swift
//

import MLX

/// protocol for environment wrappers.
/// is generic over the `InnerEnv` it wraps for better type safety.
public protocol Wrapper: Env where
    Observation == InnerEnv.Observation,
    Action == InnerEnv.Action,
    ObservationSpace == InnerEnv.ObservationSpace,
    ActionSpace == InnerEnv.ActionSpace
{
    associatedtype InnerEnv: Env
    
    /// "inner" environment instance.
    var env: InnerEnv { get set }
    
    /// requires that all wrappers can be initialized with the
    /// environment they are wrapping.
    init(env: InnerEnv)
}

/// This extension provides the "pass-through" logic. By default,
/// a Wrapper just forwards all calls and properties to its
/// inner `env`.
public extension Wrapper {
    var action_space: InnerEnv.ActionSpace {
        return env.action_space
    }

    var observation_space: InnerEnv.ObservationSpace {
        return env.observation_space
    }

    var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }
    
    var render_mode: String? {
        get { env.render_mode }
        set { env.render_mode = newValue }
    }

    var unwrapped: any Env {
        return env.unwrapped
    }
    
    mutating func step(_ action: Action) -> StepResult {
        return env.step(action)
    }

    mutating func reset(
        seed: UInt64?,
        options: [String: Any]?
    ) -> ResetResult {
        return env.reset(seed: seed, options: options)
    }

    func close() {
        return env.close()
    }

    func render() {
        env.render()
    }
}