//
//  DQNPolicy.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Q-Network for DQN that outputs Q-values for all discrete actions.
public final class DQNPolicy: Module, Policy {
    public let observationSpace: any Space<MLXArray>
    public let actionSpace: any Space
    public let normalizeImages: Bool
    public let netArch: [Int]
    public let nActions: Int

    private let _featuresExtractor: any FeaturesExtractor

    public var featuresExtractor: any FeaturesExtractor { _featuresExtractor }

    @ModuleInfo private var qNet: Sequential

    /// Creates a DQN policy.
    ///
    /// - Parameters:
    ///   - observationSpace: The observation space.
    ///   - nActions: Number of discrete actions.
    ///   - config: Configuration for the Q-Network.
    public init(
        observationSpace: any Space<MLXArray>,
        nActions: Int,
        config: DQNPolicyConfig = DQNPolicyConfig()
    ) {
        self.observationSpace = observationSpace
        self.actionSpace = Discrete(n: nActions)
        self.normalizeImages = config.normalizeImages
        self.netArch = config.netArch
        self.nActions = nActions

        let extractor = config.featuresExtractor.make(
            observationSpace: observationSpace,
            normalizeImages: config.normalizeImages
        )
        self._featuresExtractor = extractor

        let featureDim = extractor.featuresDim

        self.qNet = MLPFactory.make(
            inputDim: featureDim,
            outputDim: nActions,
            hiddenLayers: config.netArch,
            activation: { config.activation.make() }
        )

        super.init()
    }

    /// Creates a DQN policy from a Discrete action space.
    ///
    /// - Parameters:
    ///   - observationSpace: The observation space.
    ///   - actionSpace: The discrete action space.
    ///   - config: Configuration for the Q-Network.
    public convenience init(
        observationSpace: any Space<MLXArray>,
        actionSpace: Discrete,
        config: DQNPolicyConfig = DQNPolicyConfig()
    ) {
        self.init(
            observationSpace: observationSpace,
            nActions: actionSpace.n,
            config: config
        )
    }

    /// Creates a DQN policy with explicit parameters.
    ///
    /// - Parameters:
    ///   - observationSpace: The observation space.
    ///   - nActions: Number of discrete actions.
    ///   - netArch: Hidden layer sizes.
    ///   - featuresExtractor: The features extractor to use.
    ///   - normalizeImages: Whether to normalize images.
    ///   - activation: Activation function factory.
    public init(
        observationSpace: any Space<MLXArray>,
        nActions: Int,
        netArch: [Int] = [64, 64],
        featuresExtractor: (any FeaturesExtractor)? = nil,
        normalizeImages: Bool = true,
        activation: @escaping () -> any UnaryLayer = { ReLU() }
    ) {
        self.observationSpace = observationSpace
        self.actionSpace = Discrete(n: nActions)
        self.normalizeImages = normalizeImages
        self.netArch = netArch
        self.nActions = nActions

        let extractor =
            featuresExtractor
            ?? FeaturesExtractorConfig.auto.make(
                observationSpace: observationSpace,
                normalizeImages: normalizeImages
            )
        self._featuresExtractor = extractor

        let featureDim = extractor.featuresDim

        self.qNet = MLPFactory.make(
            inputDim: featureDim,
            outputDim: nActions,
            hiddenLayers: netArch,
            activation: activation
        )

        super.init()
    }

    public func makeFeatureExtractor() -> any FeaturesExtractor {
        FlattenExtractor(featuresDim: observationSpace.shape?.reduce(1, *) ?? 1)
    }

    public func predictInternal(observation: MLXArray, deterministic: Bool) -> MLXArray {
        let qValues = qNet(observation)
        return MLX.argMax(qValues, axis: -1)
    }

    /// Computes Q-values for all actions given an observation.
    ///
    /// - Parameter obs: The observation tensor.
    /// - Returns: Q-values for all actions with shape [batch, nActions].
    public func forward(obs: MLXArray) -> MLXArray {
        let features = extractFeatures(obs: obs, featuresExtractor: featuresExtractor)
        return qNet(features)
    }

    /// Computes Q-values from a Dict observation.
    ///
    /// - Parameter obs: The Dict observation.
    /// - Returns: Q-values for all actions.
    public func forward(obs: [String: MLXArray]) -> MLXArray {
        guard let dictExtractor = featuresExtractor as? any DictFeaturesExtractor else {
            preconditionFailure("forward(obs: [String: MLXArray]) requires a DictFeaturesExtractor")
        }
        let features = extractFeatures(obs: obs, featuresExtractor: dictExtractor)
        return qNet(features)
    }

    /// Predicts the action with the highest Q-value.
    ///
    /// - Parameters:
    ///   - observation: The observation tensor.
    ///   - deterministic: Ignored for DQN (always deterministic from Q-values).
    /// - Returns: The action index as an MLXArray.
    public func predict(observation: MLXArray, deterministic: Bool = true) -> MLXArray {
        setTrainingMode(false)
        let qValues = forward(obs: observation)
        return MLX.argMax(qValues, axis: -1)
    }

    /// Predicts the action from a Dict observation.
    ///
    /// - Parameters:
    ///   - observation: The Dict observation.
    ///   - deterministic: Ignored for DQN.
    /// - Returns: The action index as an MLXArray.
    public func predict(observation: [String: MLXArray], deterministic: Bool = true) -> MLXArray {
        setTrainingMode(false)
        let qValues = forward(obs: observation)
        return MLX.argMax(qValues, axis: -1)
    }
}
