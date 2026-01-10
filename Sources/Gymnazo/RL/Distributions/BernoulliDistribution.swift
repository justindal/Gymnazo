//
//  BernoulliDistribution.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Bernoulli distribution for binary action spaces.
///
/// Used for multi-binary action spaces where each dimension is independent
/// and can be 0 or 1. The distribution is parameterized by logits.
public final class BernoulliDistribution: Distribution, DistributionWithNet {
    private let actionDim: Int
    private var logits: MLXArray
    private var probs: MLXArray
    
    /// Creates a BernoulliDistribution.
    ///
    /// - Parameter actionDim: Number of binary actions.
    public init(actionDim: Int) {
        self.actionDim = actionDim
        self.logits = MLXArray([])
        self.probs = MLXArray([])
    }
    
    /// Creates the network for producing action logits.
    ///
    /// - Parameters:
    ///   - latentDim: Dimension of the latent features.
    ///   - logStdInit: Unused for Bernoulli (required by protocol).
    /// - Returns: Action network and nil (no log_std for Bernoulli).
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
    ///   - actionDim: Number of binary actions.
    /// - Returns: Action network.
    public static func probaDistributionNet(
        latentDim: Int,
        actionDim: Int
    ) -> Linear {
        return Linear(latentDim, actionDim)
    }
    
    /// Sets the distribution parameters from logits.
    ///
    /// - Parameter actionLogits: Unnormalized log probabilities.
    /// - Returns: Self for chaining.
    @discardableResult
    public func probaDistribution(actionLogits: MLXArray) -> Self {
        self.logits = actionLogits
        self.probs = MLX.sigmoid(actionLogits)
        return self
    }
    
    public func logProb(_ actions: MLXArray) -> MLXArray {
        let actionsFloat = actions.asType(.float32)
        let logP1 = -MLX.log(1.0 + MLX.exp(-logits))
        let logP0 = -MLX.log(1.0 + MLX.exp(logits))
        let logP = actionsFloat * logP1 + (1.0 - actionsFloat) * logP0
        return MLX.sum(logP, axis: -1)
    }
    
    public func entropy() -> MLXArray? {
        let p = probs
        let entropy = -(p * MLX.log(p + 1e-10) + (1.0 - p) * MLX.log(1.0 - p + 1e-10))
        return MLX.sum(entropy, axis: -1)
    }
    
    public func sample(key: MLXArray? = nil) -> MLXArray {
        if let key = key {
            return MLX.bernoulli(probs, key: key).asType(.float32)
        }
        return MLX.bernoulli(probs).asType(.float32)
    }
    
    public func mode() -> MLXArray {
        return MLX.where(probs .>= 0.5, MLXArray(1.0), MLXArray(0.0))
    }
}

