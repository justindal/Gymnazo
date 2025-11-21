//
//  NormalizeObservation.swift
//

import Foundation
import MLX

/// normalize observation to mean 0 and variance 1.
/// this wrapper maintains a running mean and variance of the observations.
public struct NormalizeObservation<InnerEnv: Environment>: Wrapper where InnerEnv.Observation == MLXArray {
    public typealias InnerEnv = InnerEnv
    
    public var env: InnerEnv
    public let epsilon: Float = 1e-8
    
    private class RunningMeanStd {
        var mean: MLXArray
        var var_sum: MLXArray
        var count: Float
        
        init(shape: [Int]) {
            self.mean = MLX.zeros(shape)
            self.var_sum = MLX.zeros(shape)
            self.count = 0
        }
        
        func update(_ x: MLXArray) {
            let batchMean = x
            count += 1
            let delta = x - mean
            mean = mean + delta / count
            let delta2 = x - mean
            var_sum = var_sum + delta * delta2
        }
        
        var variance: MLXArray {
            if count < 2 { return MLX.ones(mean.shape) }
            return var_sum / (count - 1)
        }
        
        var std: MLXArray {
            return MLX.sqrt(variance)
        }
    }
    
    private let rms: RunningMeanStd
    
    public init(env: InnerEnv) {
        self.env = env
        guard let shape = env.observation_space.shape else {
            fatalError("NormalizeObservation requires an observation space with a defined shape")
        }
        self.rms = RunningMeanStd(shape: shape)
    }
    
    public func step(_ action: InnerEnv.Action) -> StepResult {
        let result = env.step(action)
        rms.update(result.obs)
        let normalized_obs = (result.obs - rms.mean) / (rms.std + epsilon)
        
        return (
            obs: normalized_obs,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
    
    public mutating func reset(seed: UInt64?, options: [String : Any]?) -> ResetResult {
        let result = env.reset(seed: seed, options: options)
        rms.update(result.obs)
        let normalized_obs = (result.obs - rms.mean) / (rms.std + epsilon)
        
        return (
            obs: normalized_obs,
            info: result.info
        )
    }
}
