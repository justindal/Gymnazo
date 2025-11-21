//
//  TransformObservation.swift
//

import Foundation
import MLX

public struct TransformObservation<InnerEnv: Environment>: Wrapper {
    public typealias InnerEnv = InnerEnv
    
    public var env: InnerEnv
    public let transform: (InnerEnv.Observation) -> InnerEnv.Observation
    public let observation_space: InnerEnv.ObservationSpace
    
    public init(env: InnerEnv, transform: @escaping (InnerEnv.Observation) -> InnerEnv.Observation, observation_space: InnerEnv.ObservationSpace? = nil) {
        self.env = env
        self.transform = transform
        self.observation_space = observation_space ?? env.observation_space
    }
    
    public init(env: InnerEnv) {
        fatalError("Must provide transform function")
    }
    
    public func step(_ action: InnerEnv.Action) -> StepResult {
        let result = env.step(action)
        return (
            obs: transform(result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
    
    public mutating func reset(seed: UInt64?, options: [String : Any]?) -> ResetResult {
        let result = env.reset(seed: seed, options: options)
        return (
            obs: transform(result.obs),
            info: result.info
        )
    }
}
