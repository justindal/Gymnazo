//
// TimeLimit.swift
//

/// wrapper that enforces a maximum number of steps per episode by emitting the
/// truncation signal once the configured limit is reached.
public final class TimeLimit<InnerEnv: Env>: Wrapper {
    public typealias Observation = InnerEnv.Observation
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = InnerEnv.ObservationSpace
    public typealias ActionSpace = InnerEnv.ActionSpace

    public var env: InnerEnv

    private let maxEpisodeSteps: Int
    private var elapsedSteps: Int = 0
    private var cachedSpec: EnvSpec?

    /// initializer that pulls the episode limit from the wrapped
    /// environment's spec (if present).
    public required convenience init(env: InnerEnv) {
        guard let maxSteps = env.spec?.maxEpisodeSteps else {
            fatalError("TimeLimit requires either an explicit maxEpisodeSteps or one defined on the wrapped environment's spec.")
        }
        self.init(env: env, maxEpisodeSteps: maxSteps)
    }

    public init(env: InnerEnv, maxEpisodeSteps: Int) {
        precondition(maxEpisodeSteps > 0, "maxEpisodeSteps must be positive, got \(maxEpisodeSteps)")
        self.env = env
        self.maxEpisodeSteps = maxEpisodeSteps
    }

    /// resets the wrapped environment and clears the elapsed step counter.
    public func reset(seed: UInt64?, options: [String : Any]?) -> Reset<Observation> {
        elapsedSteps = 0
        cachedSpec = nil
        return env.reset(seed: seed, options: options)
    }

    /// steps the wrapped environment and converts the step into a truncation when the
    /// elapsed step counter hits the configured limit.
    public func step(_ action: Action) -> Step<Observation> {
        let result = env.step(action)
        elapsedSteps += 1

        let truncated = result.truncated || (elapsedSteps >= maxEpisodeSteps)

        return Step(
            obs: result.obs,
            reward: result.reward,
            terminated: result.terminated,
            truncated: truncated,
            info: result.info
        )
    }

    /// spec advertises the enforced episode limit.
    public var spec: EnvSpec? {
        get {
            if let cachedSpec: EnvSpec {
                return cachedSpec
            }

            guard var envSpec: EnvSpec = env.spec else {
                return nil
            }

            envSpec.maxEpisodeSteps = maxEpisodeSteps
            cachedSpec = envSpec
            return envSpec
        }
        set {
            cachedSpec = nil
            env.spec = newValue
        }
    }
}
