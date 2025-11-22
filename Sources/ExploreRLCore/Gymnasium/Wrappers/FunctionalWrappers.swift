//
// FunctionalWrappers.swift
//

open class ObservationWrapper<InnerEnv: Env>: Wrapper {
    public var env: InnerEnv

    public required init(env: InnerEnv) {
        self.env = env
    }

    /// Override to transform observations returned from the inner environment.
    open func observation(_ observation: Observation) -> Observation {
        fatalError("observation(_:) must be overridden in a subclass")
    }

    public func reset(seed: UInt64?, options: [String : Any]?) -> (obs: Observation, info: [String: Any]) {
        let result: (obs: InnerEnv.Observation, info: [String : Any]) = env.reset(seed: seed, options: options)
        return (obs: observation(result.obs), info: result.info)
    }

    public func step(_ action: Action) -> (
        obs: Observation,
        reward: Double,
        terminated: Bool,
        truncated: Bool,
        info: [String: Any]
    ) {
        let result: (obs: InnerEnv.Observation, reward: Double, terminated: Bool, truncated: Bool, info: [String : Any]) = env.step(action)
        return (
            obs: observation(result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
}

open class RewardWrapper<InnerEnv: Env>: Wrapper {
    public var env: InnerEnv

    public required init(env: InnerEnv) {
        self.env = env
    }

    /// override to transform the reward emitted by the inner environment.
    open func reward(_ reward: Double) -> Double {
        fatalError("reward(_:) must be overridden in a subclass")
    }

    public func step(_ action: Action) -> (
        obs: Observation,
        reward: Double,
        terminated: Bool,
        truncated: Bool,
        info: [String: Any]
    ) {
        let result = env.step(action)
        return (
            obs: result.obs,
            reward: reward(result.reward),
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
}

open class ActionWrapper<InnerEnv: Env>: Wrapper {
    public var env: InnerEnv

    public required init(env: InnerEnv) {
        self.env = env
    }

    /// override to transform the incoming action before delegating to the inner environment.
    open func action(_ action: Action) -> Action {
        fatalError("action(_:) must be overridden in a subclass")
    }

    public func step(_ action: Action) -> (
        obs: Observation,
        reward: Double,
        terminated: Bool,
        truncated: Bool,
        info: [String: Any]
    ) {
        return env.step(self.action(action))
    }
}
