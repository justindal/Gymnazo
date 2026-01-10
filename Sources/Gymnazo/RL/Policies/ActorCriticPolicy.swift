//
//  ActorCriticPolicy.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Policy protocol for actor-critic algorithms (A2C, PPO, etc.).
///
/// Has both policy (actor) and value (critic) prediction capabilities.
///
/// - Parameters:
///     - netArch: Network architecture specification for policy and value networks.
///     - orthoInit: Whether to use orthogonal initialization.
///     - useSDE: Whether to use State-Dependent Exploration.
///     - logStdInit: Initial value for log standard deviation.
///     - fullStd: Whether to use (features x actions) parameters for std in gSDE.
///     - shareFeatureExtractor: Whether to share features extractor between actor and critic.
public protocol ActorCriticPolicy: Policy {
    var netArch: NetArch { get }
    var orthoInit: Bool { get }
    var useSDE: Bool { get }
    var logStdInit: Float { get }
    var fullStd: Bool { get }
    var shareFeatureExtractor: Bool { get }

    var featuresDim: Int { get }
    var piFeatureExtractor: any FeaturesExtractor { get }
    var vfFeatureExtractor: any FeaturesExtractor { get }

    var mlpExtractor: MLPExtractor { get }
    var actionNet: any UnaryLayer { get }
    var valueNet: Linear { get }

    var actionDist: any Distribution { get }
    var logStd: MLXArray? { get set }

    /// Forward pass through both actor and critic networks.
    ///
    /// - Parameters:
    ///   - obs: Observation tensor.
    ///   - deterministic: Whether to use deterministic actions.
    /// - Returns: Tuple of (actions, values, log_prob).
    func forward(obs: MLXArray, deterministic: Bool) -> (MLXArray, MLXArray, MLXArray)

    /// Evaluates actions given observations.
    ///
    /// - Parameters:
    ///   - obs: Observation tensor.
    ///   - actions: Actions to evaluate.
    /// - Returns: Tuple of (values, log_prob, entropy).
    func evaluateActions(obs: MLXArray, actions: MLXArray) -> (MLXArray, MLXArray, MLXArray?)

    /// Gets the current policy distribution for given observations.
    ///
    /// - Parameter obs: Observation tensor.
    /// - Returns: The action distribution.
    func getDistribution(obs: MLXArray) -> any Distribution

    /// Predicts values for given observations.
    ///
    /// - Parameter obs: Observation tensor.
    /// - Returns: Predicted values.
    func predictValues(obs: MLXArray) -> MLXArray

    /// Samples new exploration weights for gSDE.
    ///
    /// - Parameter nEnvs: Number of environments.
    func resetNoise(nEnvs: Int)
}

extension ActorCriticPolicy {
    public var orthoInit: Bool { true }
    public var useSDE: Bool { false }
    public var logStdInit: Float { 0.0 }
    public var fullStd: Bool { true }
    public var shareFeatureExtractor: Bool { true }

    public var piFeatureExtractor: any FeaturesExtractor {
        featuresExtractor
    }

    public var vfFeatureExtractor: any FeaturesExtractor {
        featuresExtractor
    }

    /// Extracts features from observations for actor and critic.
    ///
    /// - Parameter obs: Observation tensor.
    /// - Returns: Features for actor and critic (may be the same if shared).
    public func extractActorCriticFeatures(obs: MLXArray) -> (MLXArray, MLXArray) {
        if shareFeatureExtractor {
            let features = extractFeatures(obs: obs, featuresExtractor: featuresExtractor)
            return (features, features)
        } else {
            let piFeatures = extractFeatures(obs: obs, featuresExtractor: piFeatureExtractor)
            let vfFeatures = extractFeatures(obs: obs, featuresExtractor: vfFeatureExtractor)
            return (piFeatures, vfFeatures)
        }
    }

    public func predictInternal(observation: MLXArray, deterministic: Bool) -> MLXArray {
        return getDistribution(obs: observation).getActions(deterministic: deterministic)
    }

    public func resetNoise(nEnvs: Int) {
        guard useSDE, let sdeDist = actionDist as? StateDependentNoiseDistribution else {
            return
        }
        sdeDist.sampleWeights(batchSize: nEnvs)
    }
}

/// Result of a forward pass through the actor-critic networks.
public struct ActorCriticOutput {
    public let actions: MLXArray
    public let values: MLXArray
    public let logProb: MLXArray

    public init(actions: MLXArray, values: MLXArray, logProb: MLXArray) {
        self.actions = actions
        self.values = values
        self.logProb = logProb
    }
}

/// Result of action evaluation.
public struct ActionEvaluation {
    public let values: MLXArray
    public let logProb: MLXArray
    public let entropy: MLXArray?

    public init(values: MLXArray, logProb: MLXArray, entropy: MLXArray?) {
        self.values = values
        self.logProb = logProb
        self.entropy = entropy
    }
}

/// Standard gains for orthogonal initialization in actor-critic methods.
public enum ActorCriticInitGains {
    public static let featuresExtractor: Float = Float(2).squareRoot()
    public static let mlpExtractor: Float = Float(2).squareRoot()
    public static let actionNet: Float = 0.01
    public static let valueNet: Float = 1.0
}

/// Helper to apply orthogonal initialization to actor-critic components.
///
/// - Parameters:
///   - featuresExtractor: The features extractor module.
///   - mlpExtractor: The MLP extractor module.
///   - actionNet: The action output network.
///   - valueNet: The value output network.
///   - shareFeatureExtractor: Whether the feature extractor is shared.
///   - piFeatureExtractor: Policy feature extractor (if not shared).
///   - vfFeatureExtractor: Value feature extractor (if not shared).
public func applyActorCriticOrthoInit(
    featuresExtractor: Module,
    mlpExtractor: Module,
    actionNet: Module,
    valueNet: Module,
    shareFeatureExtractor: Bool = true,
    piFeatureExtractor: Module? = nil,
    vfFeatureExtractor: Module? = nil
) throws {
    if shareFeatureExtractor {
        try applyOrthoInit(to: featuresExtractor, gain: ActorCriticInitGains.featuresExtractor)
    } else {
        if let pi = piFeatureExtractor {
            try applyOrthoInit(to: pi, gain: ActorCriticInitGains.featuresExtractor)
        }
        if let vf = vfFeatureExtractor {
            try applyOrthoInit(to: vf, gain: ActorCriticInitGains.featuresExtractor)
        }
    }

    try applyOrthoInit(to: mlpExtractor, gain: ActorCriticInitGains.mlpExtractor)
    try applyOrthoInit(to: actionNet, gain: ActorCriticInitGains.actionNet)
    try applyOrthoInit(to: valueNet, gain: ActorCriticInitGains.valueNet)
}

/// Recursively applies orthogonal initialization to a module and its children.
private func applyOrthoInit(to module: Module, gain: Float) throws {
    try initWeightsOrthogonal(module, gain: gain)

    for (_, item) in module.children() {
        try applyOrthoInit(item: item, gain: gain)
    }
}

private func applyOrthoInit(item: NestedItem<String, Module>, gain: Float) throws {
    switch item {
    case .none:
        return
    case .value(let module):
        try applyOrthoInit(to: module, gain: gain)
    case .array(let items):
        for i in items {
            try applyOrthoInit(item: i, gain: gain)
        }
    case .dictionary(let dict):
        for (_, i) in dict {
            try applyOrthoInit(item: i, gain: gain)
        }
    }
}
