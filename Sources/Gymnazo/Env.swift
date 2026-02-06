//
//  Env.swift
//

import MLX

public protocol Env {
    var actionSpace: any Space { get }
    var observationSpace: any Space { get }
    var spec: EnvSpec? { get set }
    var renderMode: RenderMode? { get set }

    var unwrapped: any Env { get }

    mutating func step(_ action: MLXArray) throws -> Step

    mutating func reset(
        seed: UInt64?,
        options: EnvOptions?
    ) throws -> Reset

    @discardableResult
    func render() throws -> RenderOutput?

    mutating func close()
}

public struct Step {
    public var obs: MLXArray
    public var reward: Double
    public var terminated: Bool
    public var truncated: Bool
    public var info: Info

    public init(
        obs: MLXArray,
        reward: Double,
        terminated: Bool,
        truncated: Bool,
        info: Info = Info()
    ) {
        self.obs = obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
    }
}

public struct Reset {
    public var obs: MLXArray
    public var info: Info

    public init(obs: MLXArray, info: Info = Info()) {
        self.obs = obs
        self.info = info
    }
}

extension Env {
    public mutating func reset(seed: UInt64) throws -> Reset {
        try self.reset(seed: seed, options: nil)
    }

    public mutating func reset() throws -> Reset {
        try self.reset(seed: nil, options: nil)
    }

    public mutating func reset(options: EnvOptions?) throws -> Reset {
        try self.reset(seed: nil, options: options)
    }

    public var unwrapped: any Env {
        self
    }

    @discardableResult
    public func render() throws -> RenderOutput? {
        nil
    }

    public mutating func close() {}
}
