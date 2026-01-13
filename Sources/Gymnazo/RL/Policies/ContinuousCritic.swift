//
//  ContinuousCritic.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Critic network protocol for DDPG/SAC/TD3.
///
/// Represents the action-state value function (Q-value function).
/// Takes the continuous action as input concatenated with the state
/// and outputs Q(s, a).
///
/// By default creates two critic networks for clipped Q-learning (TD3/SAC).
///
/// - Parameters:
///     - observationSpace: Observation space.
///     - actionSpace: Action space (must be Box for continuous actions).
///     - netArch: Network architecture for Q-networks.
///     - featuresExtractor: Network to extract features.
///     - featuresDim: Number of features from the extractor.
///     - nCritics: Number of critic networks (default 2).
///     - shareFeaturesExtractor: Whether the extractor is shared with the actor.
public protocol ContinuousCritic: Model {
    var netArch: [Int] { get }
    var featuresDim: Int { get }
    var nCritics: Int { get }
    var shareFeaturesExtractor: Bool { get }
    var qNetworks: [Sequential] { get }
    
    /// Forward pass through all critic networks.
    ///
    /// - Parameters:
    ///   - obs: Observation tensor.
    ///   - actions: Action tensor.
    /// - Returns: Q-values from each critic network.
    func forward(obs: MLXArray, actions: MLXArray) -> [MLXArray]
    
    /// Forward pass through only the first critic network.
    ///
    /// Reduces computation when only one estimate is needed
    /// (e.g., when updating the policy in TD3).
    ///
    /// - Parameters:
    ///   - obs: Observation tensor.
    ///   - actions: Action tensor.
    /// - Returns: Q-value from the first critic.
    func q1Forward(obs: MLXArray, actions: MLXArray) -> MLXArray
}

extension ContinuousCritic {
    public var nCritics: Int { 2 }
    public var shareFeaturesExtractor: Bool { true }
    
    public func forward(obs: MLXArray, actions: MLXArray) -> [MLXArray] {
        guard let extractor = featuresExtractor else {
            preconditionFailure("ContinuousCritic requires a features extractor")
        }
        
        let features = extractFeatures(obs: obs, featuresExtractor: extractor)
        let qvalueInput = MLX.concatenated([features, actions], axis: 1)
        
        return qNetworks.map { $0(qvalueInput) }
    }
    
    public func q1Forward(obs: MLXArray, actions: MLXArray) -> MLXArray {
        guard let extractor = featuresExtractor else {
            preconditionFailure("ContinuousCritic requires a features extractor")
        }
        
        let features = extractFeatures(obs: obs, featuresExtractor: extractor)
        let qvalueInput = MLX.concatenated([features, actions], axis: 1)
        
        return qNetworks[0](qvalueInput)
    }
}

/// Creates Q-networks for continuous critics.
///
/// - Parameters:
///   - featuresDim: Dimension of features from the extractor.
///   - actionDim: Dimension of the action space.
///   - netArch: Hidden layer sizes.
///   - nCritics: Number of Q-networks to create.
///   - activation: Activation function.
/// - Returns: Array of Q-network modules.
public func createQNetworks(
    featuresDim: Int,
    actionDim: Int,
    netArch: [Int],
    nCritics: Int = 2,
    activation: @escaping () -> any UnaryLayer = { ReLU() }
) -> [Sequential] {
    let inputDim = featuresDim + actionDim
    
    return (0..<nCritics).map { _ in
        MLPFactory.make(
            inputDim: inputDim,
            outputDim: 1,
            hiddenLayers: netArch,
            activation: activation
        )
    }
}

/// Gets the action dimension from a Box action space.
///
/// - Parameter actionSpace: The action space (must be Box).
/// - Returns: The flattened action dimension.
public func getActionDim(_ actionSpace: any Space) -> Int {
    guard let box = boxSpace(from: actionSpace) else {
        preconditionFailure(
            "ContinuousCritic requires a Box action space, got \(type(of: actionSpace))"
        )
    }
    return box.shape?.reduce(1, *) ?? 1
}
