//
//  SquashedDiagGaussianDistribution.swift
//  Gymnazo
//

import MLX

/// Squashed Gaussian distribution with diagonal covariance.
///
/// Samples from a diagonal Gaussian and applies `tanh` to bound actions in `[-1, 1]`.
public final class SquashedDiagGaussianDistribution: Distribution {
    private let epsilon: Float = 1e-6

    private var mean: MLXArray
    private var logStd: MLXArray

    public init() {
        self.mean = MLXArray([])
        self.logStd = MLXArray([])
    }

    @discardableResult
    public func probaDistribution(meanActions: MLXArray, logStd: MLXArray) -> Self {
        self.mean = meanActions
        self.logStd = logStd
        return self
    }

    public func sampleAndLogProb(key: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let std = MLX.exp(logStd)
        let noise: MLXArray
        if let key = key {
            noise = MLX.normal(mean.shape, key: key)
        } else {
            noise = MLX.normal(mean.shape)
        }
        let preTanh = mean + std * noise
        let actions = MLX.tanh(preTanh)
        let logProb = logProbFromPreTanh(preTanh, squashedActions: actions)
        return (actions, logProb)
    }

    public func modeAndLogProb() -> (MLXArray, MLXArray) {
        let preTanh = mean
        let actions = MLX.tanh(preTanh)
        let logProb = logProbFromPreTanh(preTanh, squashedActions: actions)
        return (actions, logProb)
    }

    public func sample(key: MLXArray? = nil) -> MLXArray {
        let (actions, _) = sampleAndLogProb(key: key)
        return actions
    }

    public func mode() -> MLXArray {
        let (actions, _) = modeAndLogProb()
        return actions
    }

    public func logProb(_ actions: MLXArray) -> MLXArray {
        let preTanh = inverseTanh(actions)
        return logProbFromPreTanh(preTanh, squashedActions: actions)
    }

    private func logProbFromPreTanh(
        _ preTanh: MLXArray,
        squashedActions: MLXArray
    ) -> MLXArray {
        let gaussianLogProb = diagGaussianLogProb(x: preTanh, mean: mean, logStd: logStd)
        let correction = MLX.sum(
            MLX.log(1.0 - squashedActions * squashedActions + epsilon),
            axis: -1
        )
        return gaussianLogProb - correction
    }

    public func entropy() -> MLXArray? {
        nil
    }

    private func inverseTanh(_ x: MLXArray) -> MLXArray {
        let clipped = MLX.clip(x, min: -1.0 + epsilon, max: 1.0 - epsilon)
        return 0.5 * (MLX.log(1.0 + clipped) - MLX.log(1.0 - clipped))
    }

    private func diagGaussianLogProb(x: MLXArray, mean: MLXArray, logStd: MLXArray) -> MLXArray {
        let std = MLX.exp(logStd)
        let variance = std * std
        let diff = x - mean
        let logTwoPi = Float.log(2.0 * Float.pi)
        let logP = -0.5 * (diff * diff / variance + 2.0 * logStd + logTwoPi)
        return MLX.sum(logP, axis: -1)
    }
}
