//
//  TransformObservation.swift
//

import Foundation
import MLX

public struct TransformObservation<InnerEnv: Env>: Wrapper {
    public typealias InnerEnv = InnerEnv
    public typealias Observation = InnerEnv.Observation
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = InnerEnv.ObservationSpace
    public typealias ActionSpace = InnerEnv.ActionSpace
    
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
    
    public mutating func step(_ action: InnerEnv.Action) -> Step<Observation> {
        let result = env.step(action)
        return Step(
            obs: transform(result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
    
    public mutating func reset(seed: UInt64?, options: [String : Any]?) -> Reset<Observation> {
        let result = env.reset(seed: seed, options: options)
        return Reset(obs: transform(result.obs), info: result.info)
    }
}
