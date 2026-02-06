//
//  TransformObservation.swift
//

import MLX

public struct TransformObservation: Wrapper {
    public var env: any Env
    public let transform: (MLXArray) -> MLXArray
    public let observationSpace: any Space
    
    public init(
        env: any Env,
        transform: @escaping (MLXArray) -> MLXArray,
        observationSpace: (any Space)? = nil
    ) {
        self.env = env
        self.transform = transform
        self.observationSpace = observationSpace ?? env.observationSpace
    }
    
    
    public mutating func step(_ action: MLXArray) throws -> Step {
        let result = try env.step(action)
        return Step(
            obs: transform(result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        let result = try env.reset(seed: seed, options: options)
        return Reset(obs: transform(result.obs), info: result.info)
    }
}

