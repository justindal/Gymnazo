/// A reward wrapper that applies a function to every reward.
public struct TransformReward<BaseEnv: Env>: Wrapper {
    public var env: BaseEnv
    public let transform: (Double) -> Double

    /// Creates the wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap.
    ///   - transform: A function applied to every reward.
    public init(env: BaseEnv, transform: @escaping (Double) -> Double) {
        self.env = env
        self.transform = transform
    }

    public mutating func step(_ action: BaseEnv.Action) throws -> Step<BaseEnv.Observation> {
        let result = try env.step(action)
        return Step(
            obs: result.obs,
            reward: transform(result.reward),
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset<BaseEnv.Observation> {
        try env.reset(seed: seed, options: options)
    }
}
