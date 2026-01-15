/// Automatically resets an environment when an episode ends.
///
/// The wrapper sets `Step.final` with the terminal observation and autoreset
/// transition information, for correct handling of episode information.
public struct AutoReset<BaseEnv: Env>: Wrapper {
    public var env: BaseEnv
    public let mode: AutoresetMode

    private var needsReset: Bool = true

    /// Creates the wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap.
    ///   - mode: Autoreset behavior (`nextStep`, `sameStep`, or `disabled`).
    public init(env: BaseEnv, mode: AutoresetMode = .nextStep) {
        self.env = env
        self.mode = mode
    }

    public init(env: BaseEnv) {
        self.init(env: env, mode: .nextStep)
    }

    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<BaseEnv.Observation> {
        needsReset = false
        return env.reset(seed: seed, options: options)
    }

    public mutating func step(_ action: BaseEnv.Action) -> Step<BaseEnv.Observation> {
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
            return Step(
                obs: result.obs,
                reward: result.reward,
                terminated: result.terminated,
                truncated: result.truncated,
                info: result.info,
                final: EpisodeFinal(
                    terminalObservation: result.obs,
                    terminalInfo: result.info,
                    autoReset: .willResetOnNextStep
                )
            )
        }

        let terminalObs = result.obs
        let terminalInfo = result.info
        let resetResult = env.reset(seed: nil, options: nil)
        needsReset = false

        return Step(
            obs: resetResult.obs,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: resetResult.info,
            final: EpisodeFinal(
                terminalObservation: terminalObs,
                terminalInfo: terminalInfo,
                autoReset: .didReset(observation: resetResult.obs, info: resetResult.info)
            )
        )
    }
}
