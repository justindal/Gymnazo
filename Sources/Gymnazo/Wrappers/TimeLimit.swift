//
// TimeLimit.swift
//

/// wrapper that enforces a maximum number of steps per episode by emitting the
/// truncation signal once the configured limit is reached.
public final class TimeLimit<BaseEnv: Env>: Wrapper {
    public var env: BaseEnv

    private let maxEpisodeSteps: Int
    private var elapsedSteps: Int = 0
    private var cachedSpec: EnvSpec?

    public init(env: BaseEnv, maxEpisodeSteps: Int) throws {
        guard maxEpisodeSteps > 0 else {
            throw GymnazoError.invalidMaxEpisodeSteps(maxEpisodeSteps)
        }
        self.env = env
        self.maxEpisodeSteps = maxEpisodeSteps
    }

    /// resets the wrapped environment and clears the elapsed step counter.
    public func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset<BaseEnv.Observation> {
        elapsedSteps = 0
        cachedSpec = nil
        return try env.reset(seed: seed, options: options)
    }

    /// steps the wrapped environment and converts the step into a truncation when the
    /// elapsed step counter hits the configured limit.
    public func step(_ action: BaseEnv.Action) throws -> Step<BaseEnv.Observation> {
        let result = try env.step(action)
        elapsedSteps += 1
        let timeLimitTruncated = elapsedSteps >= maxEpisodeSteps
        let truncated = result.truncated || timeLimitTruncated
        var info = result.info
        if timeLimitTruncated {
            info["TimeLimit.truncated"] = true
        }
        return Step(
            obs: result.obs,
            reward: result.reward,
            terminated: result.terminated,
            truncated: truncated,
            info: info
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
