//
//  TransformObservation.swift
//

import Foundation
import MLX

public struct TransformObservation<BaseEnv: Env>: Wrapper {
    public var env: BaseEnv
    public let transform: (BaseEnv.Observation) -> BaseEnv.Observation
    public let observationSpace: BaseEnv.ObservationSpace
    
    public init(
        env: BaseEnv,
        transform: @escaping (BaseEnv.Observation) -> BaseEnv.Observation,
        observationSpace: BaseEnv.ObservationSpace? = nil
    ) {
        self.env = env
        self.transform = transform
        self.observation_space = observation_space ?? env.observation_space
    }
    
    public init(env: BaseEnv) {
        fatalError("Must provide transform function")
    }
    
    public mutating func step(_ action: BaseEnv.Action) -> Step<BaseEnv.Observation> {
        let result = env.step(action)
        return Step(
            obs: transform(result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
    
    public mutating func reset(seed: UInt64?, options: [String : Any]?) -> Reset<BaseEnv.Observation> {
        let result = env.reset(seed: seed, options: options)
        return Reset(obs: transform(result.obs), info: result.info)
    }
}
