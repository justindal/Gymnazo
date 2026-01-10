import MLX

/// A reward wrapper that applies stateful shaping based on observation and termination.
///
/// Unlike ``TransformReward`` which only transforms based on the reward value,
/// this wrapper provides access to the observation and termination status.
///
/// Example usage:
/// ```swift
/// let env = MountainCar()
///     .rewardsShaped { reward, obs, terminated in
///         let position = obs[0].item(Float.self)
///         let velocity = obs[1].item(Float.self)
///         var shaped = reward
///         shaped += Double(abs(velocity)) * 10.0
///         if terminated {
///             shaped += 100.0
///         }
///         return shaped
///     }
/// ```
public struct ShapeReward<InnerEnv: Env>: Wrapper where InnerEnv.Observation == MLXArray {
    public typealias InnerEnv = InnerEnv
    public typealias Observation = InnerEnv.Observation
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = InnerEnv.ObservationSpace
    public typealias ActionSpace = InnerEnv.ActionSpace

    public var env: InnerEnv
    
    /// A closure that receives the original reward, observation, and termination status,
    /// and returns the shaped reward.
    public let shaper: (Double, MLXArray, Bool) -> Double

    /// Creates the wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap.
    ///   - shaper: A function that transforms rewards based on (reward, observation, terminated).
    public init(env: InnerEnv, shaper: @escaping (Double, MLXArray, Bool) -> Double) {
        self.env = env
        self.shaper = shaper
    }

    public init(env: InnerEnv) {
        fatalError("Must provide shaper function")
    }

    public mutating func step(_ action: InnerEnv.Action) -> Step<Observation> {
        let result = env.step(action)
        eval(result.obs)
        let shapedReward = shaper(result.reward, result.obs, result.terminated)
        return Step(
            obs: result.obs,
            reward: shapedReward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<Observation> {
        env.reset(seed: seed, options: options)
    }
}
