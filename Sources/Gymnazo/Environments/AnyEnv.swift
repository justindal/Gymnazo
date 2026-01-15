import MLX

/// Converts an environment into an `AnyEnv`.
@inlinable
public func asAnyEnv(_ env: any Env) -> AnyEnv {
    guard let typed = env as? any Env<MLXArray, MLXArray> else {
        preconditionFailure(
            "Expected Env<MLXArray, MLXArray> but got \(type(of: env))"
        )
    }
    return AnyEnv(typed)
}

/// A type-erased `Env` wrapper.
public struct AnyEnv: Env {
    public typealias Observation = MLXArray
    public typealias Action = MLXArray
    public typealias ObservationSpace = AnySpace
    public typealias ActionSpace = AnySpace

    private var env: any Env<MLXArray, MLXArray>

    public let observationSpace: AnySpace
    public let actionSpace: AnySpace

    public var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }

    public var renderMode: String? {
        get { env.renderMode }
        set { env.renderMode = newValue }
    }

    public var unwrapped: any Env { env.unwrapped }

    public init(_ env: any Env<MLXArray, MLXArray>) {
        self.env = env

        guard let obsSpace = env.observationSpace as? any MLXSpace else {
            preconditionFailure(
                "AnyEnv requires an MLX observation space; got \(type(of: env.observationSpace))"
            )
        }
        guard let actSpace = env.actionSpace as? any MLXSpace else {
            preconditionFailure(
                "AnyEnv requires an MLX action space; got \(type(of: env.actionSpace))"
            )
        }

        self.observationSpace = AnySpace(obsSpace)
        self.actionSpace = AnySpace(actSpace)
    }

    public mutating func step(_ action: MLXArray) -> Step<MLXArray> {
        env.step(action)
    }

    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<MLXArray> {
        env.reset(seed: seed, options: options)
    }
}
