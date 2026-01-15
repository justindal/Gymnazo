//
// EnvChecker.swift
//

/// checks that mirror Gymnazo's passive environment validation utilities.
enum PassiveEnvChecks {
    static func ensureSpacesExist<E: Env>(for env: E) {
        // ensure action/observation spaces can be touched without crashing.
        _ = env.actionSpace
        _ = env.observationSpace
    }

    static func validateAction<E: Env>(_ env: E, action: E.Action) {
        if !env.actionSpace.contains(action) {
            fatalError("Action \(action) is outside the declared actionSpace for env \(env.spec?.id ?? "<unknown>")")
        }
    }

    static func validateObservation<E: Env>(_ env: E, observation: E.Observation) {
        if !env.observationSpace.contains(observation) {
            fatalError("Observation is outside the declared observationSpace for env \(env.spec?.id ?? "<unknown>")")
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
