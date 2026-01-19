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
public struct ShapeReward<BaseEnv: Env>: Wrapper where BaseEnv.Observation == MLXArray {
    public var env: BaseEnv
    
    /// A closure that receives the original reward, observation, and termination status,
    /// and returns the shaped reward.
    public let shaper: (Double, MLXArray, Bool) -> Double

    /// Creates the wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap.
    ///   - shaper: A function that transforms rewards based on (reward, observation, terminated).
    public init(env: BaseEnv, shaper: @escaping (Double, MLXArray, Bool) -> Double) {
        self.env = env
        self.shaper = shaper
    }

    public mutating func step(_ action: BaseEnv.Action) throws -> Step<BaseEnv.Observation> {
        let result = try env.step(action)
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

    public mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset<BaseEnv.Observation> {
        try env.reset(seed: seed, options: options)
    }
}
