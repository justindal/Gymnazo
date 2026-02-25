//
//  TD3.swift
//  Gymnazo
//
//  Created by Justin Daludado on 2026-02-24.
//

import MLX

protocol TD3ActionNoise: Sendable {
    func reset()
    func sample(shape: [Int], key: MLXArray?) -> MLXArray
}

final class TD3NormalActionNoise: TD3ActionNoise, @unchecked Sendable {
    private let std: Float

    init(std: Float) {
        self.std = max(0.0, std)
    }

    func reset() {}

    func sample(shape: [Int], key: MLXArray?) -> MLXArray {
        if let key {
            return MLX.normal(shape, key: key) * std
        }
        return MLX.normal(shape) * std
    }
}

final class TD3OrnsteinUhlenbeckActionNoise: TD3ActionNoise, @unchecked Sendable {
    private let std: Float
    private let theta: Float
    private let dt: Float
    private let initialNoise: Float

    private var previousNoise: MLXArray?

    init(
        std: Float,
        theta: Float,
        dt: Float,
        initialNoise: Float
    ) {
        self.std = max(0.0, std)
        self.theta = max(0.0, theta)
        self.dt = max(1e-9, dt)
        self.initialNoise = initialNoise
        self.previousNoise = nil
    }

    func reset() {
        previousNoise = nil
    }

    func sample(shape: [Int], key: MLXArray?) -> MLXArray {
        let prev = previousNoise ?? (MLX.zeros(shape) + initialNoise)
        let gaussian: MLXArray
        if let key {
            gaussian = MLX.normal(shape, key: key)
        } else {
            gaussian = MLX.normal(shape)
        }

        let drift = theta * dt
        let sigma = std * Float(Double(dt).squareRoot())
        let noise =
            prev
            + drift * (MLX.zeros(like: prev) - prev)
            + sigma * gaussian
        let nextNoise = MLX.stopGradient(noise)
        eval(nextNoise)
        previousNoise = nextNoise
        return nextNoise
    }
}
