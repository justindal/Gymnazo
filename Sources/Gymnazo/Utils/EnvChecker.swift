//
// EnvChecker.swift
//

/// checks that mirror Gymnazo's passive environment validation utilities.
enum PassiveEnvChecks {
    static func ensureSpacesExist<E: Env>(for env: E) {
        // ensure action/observation spaces can be touched without crashing.
        _ = env.action_space
        _ = env.observation_space
    }

    static func validateAction<E: Env>(_ env: E, action: E.Action) {
        if !env.action_space.contains(action) {
            fatalError("Action \(action) is outside the declared action_space for env \(env.spec?.id ?? "<unknown>")")
        }
    }

    static func validateObservation<E: Env>(_ env: E, observation: E.Observation) {
        if !env.observation_space.contains(observation) {
            fatalError("Observation is outside the declared observation_space for env \(env.spec?.id ?? "<unknown>")")
        }
    }

    static func step<E: Env>(env: inout E, action: E.Action) -> Step<E.Observation> {
        validateAction(env, action: action)
        let result = env.step(action)
        validateObservation(env, observation: result.obs)
        return result
    }

    static func reset<E: Env>(env: inout E, seed: UInt64?, options: [String: Any]?) -> Reset<E.Observation> {
        let result = env.reset(seed: seed, options: options)
        validateObservation(env, observation: result.obs)
        return result
    }

    static func render<E: Env>(env: inout E) {
        env.render()
    }
}
