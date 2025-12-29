/// Automatically resets an environment when an episode ends.
///
/// When autoreset happens, the wrapper stores the terminated episode’s final transition under:
/// - `info["final_observation"]`
/// - `info["final_info"]`
public struct AutoReset<InnerEnv: Env>: Wrapper {
    public typealias InnerEnv = InnerEnv
    public typealias Observation = InnerEnv.Observation
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = InnerEnv.ObservationSpace
    public typealias ActionSpace = InnerEnv.ActionSpace

    public var env: InnerEnv
    public let mode: AutoresetMode

    private var needsReset: Bool = true

    /// Creates the wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap.
    ///   - mode: Autoreset behavior (`nextStep`, `sameStep`, or `disabled`).
    public init(env: InnerEnv, mode: AutoresetMode = .nextStep) {
        self.env = env
        self.mode = mode
    }

    public init(env: InnerEnv) {
        self.init(env: env, mode: .nextStep)
    }

    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<Observation> {
        needsReset = false
        return env.reset(seed: seed, options: options)
    }

    public mutating func step(_ action: InnerEnv.Action) -> Step<Observation> {
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
                final: Final(obs: result.obs, info: result.info)
            )
        }

        let finalObs = result.obs
        let resetResult = env.reset(seed: nil, options: nil)
        needsReset = false

        return Step(
            obs: resetResult.obs,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: resetResult.info,
            final: Final(obs: finalObs, info: result.info)
        )
    }
}

