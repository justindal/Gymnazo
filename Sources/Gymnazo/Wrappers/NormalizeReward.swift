import Foundation

/// Normalizes immediate rewards using a running estimate of return variance.
public struct NormalizeReward<BaseEnv: Env>: Wrapper {
    public var env: BaseEnv
    public let gamma: Double
    public let epsilon: Double

    private let rms = RunningMeanStd<Double>()
    private var returns: Double = 0

    /// Creates the wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap.
    ///   - gamma: Discount factor used to compute the running return.
    ///   - epsilon: Numerical stability constant.
    public init(env: BaseEnv, gamma: Double = 0.99, epsilon: Double = 1e-8) {
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
    }

    public init(env: BaseEnv) {
        self.init(env: env, gamma: 0.99, epsilon: 1e-8)
    }

    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<BaseEnv.Observation> {
        returns = 0
        return env.reset(seed: seed, options: options)
    }

    public mutating func step(_ action: BaseEnv.Action) -> Step<BaseEnv.Observation> {
        let result = env.step(action)

        returns = gamma * returns + result.reward
        rms.update(returns)

        let denom = Foundation.sqrt(rms.variance + epsilon)
        let normalized = result.reward / denom

        if result.terminated || result.truncated {
            returns = 0
        }

        return Step(
            obs: result.obs,
            reward: normalized,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
}

