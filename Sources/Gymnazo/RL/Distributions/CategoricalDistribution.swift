//
//  CategoricalDistribution.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Categorical distribution for discrete action spaces.
///
/// Used when the action space is a finite set of discrete choices.
/// The distribution is parameterized by logits (unnormalized log probabilities).
public final class CategoricalDistribution: Distribution, DistributionWithNet {
    private let actionDim: Int
    private var logits: MLXArray
    private var probs: MLXArray
    
    /// Creates a CategoricalDistribution.
    ///
    /// - Parameter actionDim: Number of discrete actions.
    public init(actionDim: Int) {
        self.actionDim = actionDim
        self.logits = MLXArray([])
        self.probs = MLXArray([])
    }
    
    /// Creates the network for producing action logits.
    ///
    /// - Parameters:
    ///   - latentDim: Dimension of the latent features.
    ///   - logStdInit: Unused for categorical (required by protocol).
    /// - Returns: Action network and nil (no log_std for categorical).
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
    ///   - actionDim: Number of discrete actions.
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
        self.probs = MLX.softmax(actionLogits, axis: -1)
        return self
    }
    
    public func logProb(_ actions: MLXArray) -> MLXArray {
        let logProbs = logSoftmax(logits, axis: -1)
        let actionIndices = actions.asType(.int32)
        return MLX.takeAlong(logProbs, actionIndices.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
    }
    
    public func entropy() -> MLXArray? {
        let logProbs = logSoftmax(logits, axis: -1)
        let entropy = -MLX.sum(probs * logProbs, axis: -1)
        return entropy
    }
    
    public func sample() -> MLXArray {
        MLX.categorical(logits, axis: -1)
    }
    
    public func mode() -> MLXArray {
        return MLX.argMax(logits, axis: -1)
    }
}

private func logSoftmax(_ x: MLXArray, axis: Int) -> MLXArray {
    let m = MLX.max(x, axis: axis)
    let mExp = m.expandedDimensions(axis: axis)
    let shifted = x - mExp
    let logZ = MLX.log(MLX.sum(MLX.exp(shifted), axis: axis)) + m
    return x - logZ.expandedDimensions(axis: axis)
}

/// Multi-categorical distribution for multi-discrete action spaces.
///
/// Used when the action space consists of multiple independent discrete choices.
public final class MultiCategoricalDistribution: Distribution {
    private let actionDims: [Int]
    private var distributions: [CategoricalDistribution]
    
    /// Creates a MultiCategoricalDistribution.
    ///
    /// - Parameter actionDims: Array of sizes for each discrete dimension.
    public init(actionDims: [Int]) {
        self.actionDims = actionDims
        self.distributions = actionDims.map { CategoricalDistribution(actionDim: $0) }
    }
    
    /// Creates the network for producing all action logits.
    ///
    /// - Parameters:
    ///   - latentDim: Dimension of the latent features.
    ///   - actionDims: Array of sizes for each discrete dimension.
    /// - Returns: Action network producing concatenated logits.
    public static func probaDistributionNet(
        latentDim: Int,
        actionDims: [Int]
    ) -> Linear {
        let totalDim = actionDims.reduce(0, +)
        return Linear(latentDim, totalDim)
    }
    
    /// Sets the distribution parameters from concatenated logits.
    ///
    /// - Parameter actionLogits: Flattened logits for all dimensions.
    /// - Returns: Self for chaining.
    @discardableResult
    public func probaDistribution(actionLogits: MLXArray) -> Self {
        var offset = 0
        for (i, dim) in actionDims.enumerated() {
            let sliceLogits = actionLogits[.ellipsis, offset..<(offset + dim)]
            distributions[i].probaDistribution(actionLogits: sliceLogits)
            offset += dim
        }
        return self
    }
    
    public func logProb(_ actions: MLXArray) -> MLXArray {
        var totalLogProb = MLXArray(0.0)
        for (i, dist) in distributions.enumerated() {
            let action = actions[.ellipsis, i]
            totalLogProb = totalLogProb + dist.logProb(action)
        }
        return totalLogProb
    }
    
    public func entropy() -> MLXArray? {
        var totalEntropy = MLXArray(0.0)
        for dist in distributions {
            if let e = dist.entropy() {
                totalEntropy = totalEntropy + e
            }
        }
        return totalEntropy
    }
    
    public func sample() -> MLXArray {
        let samples = distributions.map { $0.sample() }
        return MLX.stacked(samples, axis: -1)
    }
    
    public func mode() -> MLXArray {
        let modes = distributions.map { $0.mode() }
        return MLX.stacked(modes, axis: -1)
    }
}

