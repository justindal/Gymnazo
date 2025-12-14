public struct AutoReset<InnerEnv: Env>: Wrapper {
    public typealias InnerEnv = InnerEnv
    public typealias Observation = InnerEnv.Observation
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = InnerEnv.ObservationSpace
    public typealias ActionSpace = InnerEnv.ActionSpace

    public var env: InnerEnv
    public let mode: AutoresetMode

    private var needsReset: Bool = true

    public init(env: InnerEnv, mode: AutoresetMode = .nextStep) {
        self.env = env
        self.mode = mode
    }

    public init(env: InnerEnv) {
        self.init(env: env, mode: .nextStep)
    }

    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> ResetResult {
        needsReset = false
        return env.reset(seed: seed, options: options)
    }

    public mutating func step(_ action: InnerEnv.Action) -> StepResult {
        if needsReset && mode == .nextStep {
            _ = env.reset(seed: nil, options: nil)
            needsReset = false
        }

        let result = env.step(action)
        let done = result.terminated || result.truncated

        if !done {
            return result
        }

        if mode == .disabled {
            return result
        }

        if mode == .nextStep {
            needsReset = true
            let baseInfo = result.info
            var info = baseInfo
            info["final_observation"] = result.obs
            info["final_info"] = baseInfo
            return (
                obs: result.obs,
                reward: result.reward,
                terminated: result.terminated,
                truncated: result.truncated,
                info: info
            )
        }

        let baseInfo = result.info
        let finalObs = result.obs
        let resetResult = env.reset(seed: nil, options: nil)
        needsReset = false

        var info = resetResult.info
        info["final_observation"] = finalObs
        info["final_info"] = baseInfo

        return (
            obs: resetResult.obs,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: info
        )
    }
}

