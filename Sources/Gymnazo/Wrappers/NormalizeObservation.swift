//
//  NormalizeObservation.swift
//

import Foundation
import MLX

/// Normalizes observations to approximately mean 0 and variance 1.
///
/// Uses Welford's online algorithm to maintain running statistics.
public struct NormalizeObservation: Wrapper {
    public var env: any Env
    public let epsilon: Float = 1e-8
    
    private let rms: RunningMeanStdMLX
    
    public init(env: any Env) throws {
        self.env = env
        guard let shape = env.observationSpace.shape else {
            throw GymnazoError.invalidObservationSpace
        }
        self.rms = RunningMeanStdMLX(shape: shape)
    }
    
    public mutating func step(_ action: MLXArray) throws -> Step {
        let result = try env.step(action)
        rms.update(result.obs)
        let normalized_obs = (result.obs - rms.mean) / (rms.std + epsilon)
        
        return Step(
            obs: normalized_obs,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
    
    public mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        let result = try env.reset(seed: seed, options: options)
        rms.update(result.obs)
        let normalized_obs = (result.obs - rms.mean) / (rms.std + epsilon)
        
        return Reset(obs: normalized_obs, info: result.info)
    }
}


