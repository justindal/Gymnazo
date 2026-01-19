//
// Wrapper.swift
//

import MLX

/// protocol for environment wrappers.
/// is generic over the `BaseEnv` it wraps for better type safety.
public protocol Wrapper<BaseEnv>: Env
where
    Observation == BaseEnv.Observation,
    Action == BaseEnv.Action
{
    associatedtype BaseEnv: Env

    /// "inner" environment instance.
    var env: BaseEnv { get set }
}

/// This extension provides the "pass-through" logic. By default,
/// a Wrapper just forwards all calls and properties to its
/// inner `env`.
extension Wrapper {
    public var actionSpace: any Space<Action> {
        env.actionSpace
    }

    public var observationSpace: any Space<Observation> {
        env.observationSpace
    }

    public var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }

    public var renderMode: RenderMode? {
        get { env.renderMode }
        set { env.renderMode = newValue }
    }

    public var unwrapped: any Env {
        env.unwrapped
    }

    public mutating func step(_ action: Action) throws -> Step<Observation> {
        try env.step(action)
    }

    public mutating func reset(
        seed: UInt64?,
        options: EnvOptions?
    ) throws -> Reset<Observation> {
        try env.reset(seed: seed, options: options)
    }

    public mutating func close() {
        env.close()
    }

    @discardableResult
    public func render() throws -> RenderOutput? {
        try env.render()
    }
}
