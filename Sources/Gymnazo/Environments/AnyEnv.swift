import MLX

/// A type-erased `Env` wrapper.
public struct AnyEnv: Env {
    public typealias Observation = MLXArray
    public typealias Action = MLXArray
    public typealias ObservationSpace = AnySpace
    public typealias ActionSpace = AnySpace

    private var env: any Env<MLXArray, MLXArray>

    public let observation_space: AnySpace
    public let action_space: AnySpace

    public var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }

    public var render_mode: String? {
        get { env.render_mode }
        set { env.render_mode = newValue }
    }

    public var unwrapped: any Env { env.unwrapped }

    public init(_ env: any Env<MLXArray, MLXArray>) {
        self.env = env

        guard let obsSpace = env.observation_space as? any MLXSpace else {
            preconditionFailure(
                "AnyEnv requires an MLX observation space; got \(type(of: env.observation_space))"
            )
        }
        guard let actSpace = env.action_space as? any MLXSpace else {
            preconditionFailure(
                "AnyEnv requires an MLX action space; got \(type(of: env.action_space))"
            )
        }

        self.observation_space = AnySpace(obsSpace)
        self.action_space = AnySpace(actSpace)
    }

    public mutating func step(_ action: MLXArray) -> Step<MLXArray> {
        env.step(action)
    }

    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<MLXArray> {
        env.reset(seed: seed, options: options)
    }
}
