//
// FunctionalWrappers.swift
//

import MLX

open class ObservationWrapper: Wrapper {
    public var env: any Env

    public required init(env: any Env) {
        self.env = env
    }

    /// Override to transform observations returned from the inner environment.
    open func observation(_ observation: MLXArray) -> MLXArray {
        observation
    }

    public func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        let result = try env.reset(seed: seed, options: options)
        return Reset(obs: observation(result.obs), info: result.info)
    }

    public func step(_ action: MLXArray) throws -> Step {
        let result = try env.step(action)
        return Step(
            obs: observation(result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
}

open class RewardWrapper: Wrapper {
    public var env: any Env

    public required init(env: any Env) {
        self.env = env
    }

    /// override to transform the reward emitted by the inner environment.
    open func reward(_ reward: Double) -> Double {
        reward
    }

    public func step(_ action: MLXArray) throws -> Step {
        let result = try env.step(action)
        return Step(
            obs: result.obs,
            reward: reward(result.reward),
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
}

open class ActionWrapper: Wrapper {
    public var env: any Env

    public required init(env: any Env) {
        self.env = env
    }

    /// override to transform the incoming action before delegating to the inner environment.
    open func action(_ action: MLXArray) -> MLXArray {
        action
    }

    public func step(_ action: MLXArray) throws -> Step {
        try env.step(self.action(action))
    }
}

