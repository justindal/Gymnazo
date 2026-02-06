//
// Wrapper.swift
//

import MLX

public protocol Wrapper: Env {
    var env: any Env { get set }
}

extension Wrapper {
    public var actionSpace: any Space {
        env.actionSpace
    }

    public var observationSpace: any Space {
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

    public mutating func step(_ action: MLXArray) throws -> Step {
        try env.step(action)
    }

    public mutating func reset(
        seed: UInt64?,
        options: EnvOptions?
    ) throws -> Reset {
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
