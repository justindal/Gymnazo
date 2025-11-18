//
// EnvChecker.swift
//

/// checks that mirror Gymnasium's passive environment validation utilities.
enum PassiveEnvChecks {
    static func ensureSpacesExist<Env: Environment>(for env: Env) {
        // ensure action/observation spaces can be touched without crashing.
        _ = env.action_space
        _ = env.observation_space
    }

    static func validateAction<Env: Environment>(_ env: Env, action: Env.Action) {
        if !env.action_space.contains(action) {
            fatalError("Action \(action) is outside the declared action_space for env \(env.spec?.id ?? "<unknown>")")
        }
    }

    static func validateObservation<Env: Environment>(_ env: Env, observation: Env.Observation) {
        if !env.observation_space.contains(observation) {
            fatalError("Observation is outside the declared observation_space for env \(env.spec?.id ?? "<unknown>")")
        }
    }

    static func step<Env: Environment>(env: inout Env, action: Env.Action) -> Env.StepResult {
        validateAction(env, action: action)
        let result: (obs: Env.Observation, reward: Double, terminated: Bool, truncated: Bool, info: [String : Any]) = env.step(action)
        validateObservation(env, observation: result.obs)
        return result
    }

    static func reset<Env: Environment>(env: inout Env, seed: UInt64?, options: [String: Any]?) -> Env.ResetResult {
        let result: (obs: Env.Observation, info: [String : Any]) = env.reset(seed: seed, options: options)
        validateObservation(env, observation: result.obs)
        return result
    }

    static func render<Env: Environment>(env: inout Env) {
        env.render()
    }
}
