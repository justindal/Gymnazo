//
// EnvChecker.swift
//

import MLX

/// Environment validation.
enum PassiveEnvChecks {
    static func ensureSpacesExist(for env: any Env) {
        _ = env.actionSpace
        _ = env.observationSpace
    }

    static func validateAction(_ env: any Env, action: MLXArray) throws {
        if !env.actionSpace.contains(action) {
            throw GymnazoError.actionOutsideSpace(envId: env.spec?.id)
        }
    }

    static func validateObservation(_ env: any Env, observation: MLXArray) throws {
        if !env.observationSpace.contains(observation) {
            throw GymnazoError.observationOutsideSpace(envId: env.spec?.id)
        }
    }

    static func step(env: inout any Env, action: MLXArray) throws -> Step {
        try validateAction(env, action: action)
        let result = try env.step(action)
        try validateObservation(env, observation: result.obs)
        return result
    }

    static func reset(env: inout any Env, seed: UInt64?, options: EnvOptions?) throws -> Reset {
        let result = try env.reset(seed: seed, options: options)
        try validateObservation(env, observation: result.obs)
        return result
    }

    static func render(env: inout any Env) throws {
        _ = try env.render()
    }
}
