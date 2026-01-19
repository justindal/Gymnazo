//
//  TransformObservation.swift
//

import Foundation
import MLX

public struct TransformObservation<BaseEnv: Env>: Wrapper {
    public var env: BaseEnv
    public let transform: (BaseEnv.Observation) -> BaseEnv.Observation
    public let observationSpace: any Space<BaseEnv.Observation>
    
    public init(
        env: BaseEnv,
        transform: @escaping (BaseEnv.Observation) -> BaseEnv.Observation,
        observationSpace: (any Space<BaseEnv.Observation>)? = nil
    ) {
        self.env = env
        self.transform = transform
        self.observationSpace = observationSpace ?? env.observationSpace
    }
    
    
    public mutating func step(_ action: BaseEnv.Action) throws -> Step<BaseEnv.Observation> {
        let result = try env.step(action)
        return Step(
            obs: transform(result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset<BaseEnv.Observation> {
        let result = try env.reset(seed: seed, options: options)
        return Reset(obs: transform(result.obs), info: result.info)
    }
}
