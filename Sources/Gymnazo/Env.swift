//
//  Env.swift
//

public protocol Env<Observation, Action> {
    associatedtype Observation
    associatedtype Action

    var actionSpace: any Space<Action> { get }
    var observationSpace: any Space<Observation> { get }
    var spec: EnvSpec? { get set }
    var renderMode: RenderMode? { get set }

    var unwrapped: any Env { get }

    mutating func step(_ action: Action) throws -> Step<Observation>

    /// resets the environment to an initial state, returning an initial observation and info.
    /// this generates a new starting state, often with some randomness controlled by the optional seed parameter.
    mutating func reset(
        seed: UInt64?,
        options: EnvOptions?
    ) throws -> Reset<Observation>

    @discardableResult
    func render() throws -> RenderOutput?

    mutating func close()
}

public struct Step<Observation> {
    public var obs: Observation
    public var reward: Double
    public var terminated: Bool
    public var truncated: Bool
    public var info: Info

    public init(
        obs: Observation,
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

public struct Reset<Observation> {
    public var obs: Observation
    public var info: Info

    public init(obs: Observation, info: Info = Info()) {
        self.obs = obs
        self.info = info
    }
}

extension Env {
    public mutating func reset(seed: UInt64) throws -> Reset<Observation> {
        try self.reset(seed: seed, options: nil)
    }

    public mutating func reset() throws -> Reset<Observation> {
        try self.reset(seed: nil, options: nil)
    }

    public mutating func reset(options: EnvOptions?) throws -> Reset<Observation> {
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
