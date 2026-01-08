//
//  DiagGaussianDistribution.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Gaussian distribution with diagonal covariance matrix.
///
/// Used for continuous action spaces. The distribution is parameterized by
/// mean actions and a log standard deviation (shared across actions or per-action).
public final class DiagGaussianDistribution: Distribution, DistributionWithNet {
    private let actionDim: Int
    private var mean: MLXArray
    private var logStd: MLXArray
    
    /// Creates a DiagGaussianDistribution.
    ///
    /// - Parameter actionDim: Dimension of the action space.
    public init(actionDim: Int) {
        self.actionDim = actionDim
        self.mean = MLXArray([])
        self.logStd = MLXArray([])
    }
    
    /// Creates the network for producing mean actions and log_std.
    ///
    /// - Parameters:
    ///   - latentDim: Dimension of the latent features.
    ///   - logStdInit: Initial value for log standard deviation.
    /// - Returns: Action network and trainable log_std parameter.
    public static func probaDistributionNet(
        latentDim: Int,
        logStdInit: Float = 0.0
    ) -> (any UnaryLayer, MLXArray?) {
        let actionNet = Linear(latentDim, 0)
        return (actionNet, nil)
    }
    
    /// Creates the network for this distribution with the correct action dim.
    ///
    /// - Parameters:
    ///   - latentDim: Dimension of the latent features.
    ///   - actionDim: Dimension of the action space.
    ///   - logStdInit: Initial value for log standard deviation.
    /// - Returns: Action network and trainable log_std parameter.
    public static func probaDistributionNet(
        latentDim: Int,
        actionDim: Int,
        logStdInit: Float = 0.0
    ) -> (Linear, MLXArray) {
        let actionNet = Linear(latentDim, actionDim)
        let logStd = MLX.zeros([actionDim]) + logStdInit
        return (actionNet, logStd)
    }
    
    /// Sets the distribution parameters from mean actions and log_std.
    ///
    /// - Parameters:
    ///   - meanActions: Mean of the Gaussian.
    ///   - logStd: Log standard deviation.
    /// - Returns: Self for chaining.
    @discardableResult
    public func probaDistribution(meanActions: MLXArray, logStd: MLXArray) -> Self {
        self.mean = meanActions
        self.logStd = logStd
        return self
    }
    
    public func logProb(_ actions: MLXArray) -> MLXArray {
        let std = MLX.exp(logStd)
        let variance = std * std
        let logScale = logStd
        
        let diff = actions - mean
        let logP = -0.5 * (diff * diff / variance + 2.0 * logScale + Float.log(2.0 * Float.pi))
        
        return MLX.sum(logP, axis: -1)
    }
    
    public func entropy() -> MLXArray? {
        let entropy = 0.5 + 0.5 * Float.log(2.0 * Float.pi) + logStd
        return MLX.sum(entropy, axis: -1)
    }
    
    public func sample() -> MLXArray {
        let std = MLX.exp(logStd)
        let noise = MLX.normal(mean.shape)
        return mean + std * noise
    }
    
    public func mode() -> MLXArray {
        return mean
    }
}

