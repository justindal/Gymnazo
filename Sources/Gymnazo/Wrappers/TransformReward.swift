/// A reward wrapper that applies a function to every reward.
public struct TransformReward<InnerEnv: Env>: Wrapper {
    public typealias InnerEnv = InnerEnv
    public typealias Observation = InnerEnv.Observation
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = InnerEnv.ObservationSpace
    public typealias ActionSpace = InnerEnv.ActionSpace

    public var env: InnerEnv
    public let transform: (Double) -> Double

    /// Creates the wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap.
    ///   - transform: A function applied to every reward.
    public init(env: InnerEnv, transform: @escaping (Double) -> Double) {
        self.env = env
        self.transform = transform
    }

    public init(env: InnerEnv) {
        fatalError("Must provide transform function")
    }

    public mutating func step(_ action: InnerEnv.Action) -> Step<Observation> {
        let result = env.step(action)
        return Step(
            obs: result.obs,
            reward: transform(result.reward),
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<Observation> {
        env.reset(seed: seed, options: options)
    }
}

