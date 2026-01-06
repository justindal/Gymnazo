//
//  Distribution.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Protocol for action distributions.
///
/// Action distributions are used in policy gradient methods to represent
/// the stochastic policy and compute log probabilities and entropy.
public protocol Distribution {
    /// Returns the log probability of taking the given actions.
    ///
    /// - Parameter actions: The actions to evaluate.
    /// - Returns: Log probability of each action.
    func logProb(_ actions: MLXArray) -> MLXArray
    
    /// Returns the entropy of the distribution.
    ///
    /// - Returns: Entropy value, or nil if not computable.
    func entropy() -> MLXArray?
    
    /// Samples actions from the distribution.
    ///
    /// - Returns: Sampled actions.
    func sample() -> MLXArray
    
    /// Returns the most likely actions (mode of the distribution).
    ///
    /// - Returns: Deterministic actions.
    func mode() -> MLXArray
    
    /// Returns actions, either sampled or deterministic.
    ///
    /// - Parameter deterministic: If true, returns the mode; otherwise samples.
    /// - Returns: Actions tensor.
    func getActions(deterministic: Bool) -> MLXArray
}

extension Distribution {
    public func getActions(deterministic: Bool) -> MLXArray {
        if deterministic {
            return mode()
        }
        return sample()
    }
}

/// Protocol for distributions that can create their network layers.
public protocol DistributionWithNet: Distribution {
    /// Creates the network layers for the distribution.
    ///
    /// - Parameters:
    ///   - latentDim: Dimension of the latent space.
    ///   - logStdInit: Initial value for log standard deviation (if applicable).
    /// - Returns: Tuple of (action network, optional log_std parameter).
    static func probaDistributionNet(
        latentDim: Int,
        logStdInit: Float
    ) -> (any UnaryLayer, MLXArray?)
}
