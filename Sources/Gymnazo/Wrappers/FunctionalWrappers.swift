//
// FunctionalWrappers.swift
//

open class ObservationWrapper<BaseEnv: Env>: Wrapper {
    public var env: BaseEnv

    public required init(env: BaseEnv) {
        self.env = env
    }

    /// Override to transform observations returned from the inner environment.
    open func observation(_ observation: BaseEnv.Observation) -> BaseEnv.Observation {
        observation
    }

    public func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset<BaseEnv.Observation> {
        let result = try env.reset(seed: seed, options: options)
        return Reset(obs: observation(result.obs), info: result.info)
    }

    public func step(_ action: BaseEnv.Action) throws -> Step<BaseEnv.Observation> {
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

open class RewardWrapper<BaseEnv: Env>: Wrapper {
    public var env: BaseEnv

    public required init(env: BaseEnv) {
        self.env = env
    }

    /// override to transform the reward emitted by the inner environment.
    open func reward(_ reward: Double) -> Double {
        reward
    }

    public func step(_ action: BaseEnv.Action) throws -> Step<BaseEnv.Observation> {
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

open class ActionWrapper<BaseEnv: Env>: Wrapper {
    public var env: BaseEnv

    public required init(env: BaseEnv) {
        self.env = env
    }

    /// override to transform the incoming action before delegating to the inner environment.
    open func action(_ action: BaseEnv.Action) -> BaseEnv.Action {
        action
    }

    public func step(_ action: BaseEnv.Action) throws -> Step<BaseEnv.Observation> {
        try env.step(self.action(action))
    }
}
