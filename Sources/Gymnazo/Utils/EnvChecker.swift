//
// EnvChecker.swift
//

/// Environment validation.
enum PassiveEnvChecks {
    static func ensureSpacesExist<E: Env>(for env: E) {
        _ = env.actionSpace
        _ = env.observationSpace
    }

    static func validateAction<E: Env>(_ env: E, action: E.Action) throws {
        if !env.actionSpace.contains(action) {
            throw GymnazoError.actionOutsideSpace(envId: env.spec?.id)
        }
    }

    static func validateObservation<E: Env>(_ env: E, observation: E.Observation) throws {
        if !env.observationSpace.contains(observation) {
            throw GymnazoError.observationOutsideSpace(envId: env.spec?.id)
        }
    }

    static func step<E: Env>(env: inout E, action: E.Action) throws -> Step<E.Observation> {
        try validateAction(env, action: action)
        let result = try env.step(action)
        try validateObservation(env, observation: result.obs)
        return result
    }

    static func reset<E: Env>(env: inout E, seed: UInt64?, options: EnvOptions?) throws -> Reset<E.Observation> {
        let result = try env.reset(seed: seed, options: options)
        try validateObservation(env, observation: result.obs)
        return result
    }

    static func render<E: Env>(env: inout E) throws {
        _ = try env.render()
    }
}
