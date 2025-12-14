import Foundation

public struct NormalizeReward<InnerEnv: Env>: Wrapper {
    public typealias InnerEnv = InnerEnv
    public typealias Observation = InnerEnv.Observation
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = InnerEnv.ObservationSpace
    public typealias ActionSpace = InnerEnv.ActionSpace

    public var env: InnerEnv
    public let gamma: Double
    public let epsilon: Double

    private final class RunningMeanStd {
        var mean: Double = 0
        var varSum: Double = 0
        var count: Double = 0

        func update(_ x: Double) {
            count += 1
            let delta = x - mean
            mean += delta / count
            let delta2 = x - mean
            varSum += delta * delta2
        }

        var variance: Double {
            if count < 2 { return 1 }
            return varSum / (count - 1)
        }
    }

    private let rms = RunningMeanStd()
    private var returns: Double = 0

    public init(env: InnerEnv, gamma: Double = 0.99, epsilon: Double = 1e-8) {
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
    }

    public init(env: InnerEnv) {
        self.init(env: env, gamma: 0.99, epsilon: 1e-8)
    }

    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> ResetResult {
        returns = 0
        return env.reset(seed: seed, options: options)
    }

    public mutating func step(_ action: InnerEnv.Action) -> StepResult {
        let result = env.step(action)

        returns = gamma * returns + result.reward
        rms.update(returns)

        let denom = Foundation.sqrt(rms.variance + epsilon)
        let normalized = result.reward / denom

        if result.terminated || result.truncated {
            returns = 0
        }

        return (
            obs: result.obs,
            reward: normalized,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
}

