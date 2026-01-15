//
//  NormalizeObservation.swift
//

import Foundation
import MLX

/// Normalizes observations to approximately mean 0 and variance 1.
///
/// Uses Welford's online algorithm to maintain running statistics.
public struct NormalizeObservation<BaseEnv: Env>: Wrapper where BaseEnv.Observation == MLXArray {
    public var env: BaseEnv
    public let epsilon: Float = 1e-8
    
    private let rms: RunningMeanStdMLX
    
    public init(env: BaseEnv) {
        self.env = env
        guard let shape = env.observation_space.shape else {
            fatalError("NormalizeObservation requires an observation space with a defined shape")
        }
        self.rms = RunningMeanStdMLX(shape: shape)
    }
    
    public mutating func step(_ action: BaseEnv.Action) -> Step<BaseEnv.Observation> {
        let result = env.step(action)
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
    
    public mutating func reset(seed: UInt64?, options: [String : Any]?) -> Reset<BaseEnv.Observation> {
        let result = env.reset(seed: seed, options: options)
        rms.update(result.obs)
        let normalized_obs = (result.obs - rms.mean) / (rms.std + epsilon)
        
        return Reset(obs: normalized_obs, info: result.info)
    }
}

