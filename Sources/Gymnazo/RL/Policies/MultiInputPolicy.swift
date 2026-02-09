//
//  MultiInputPolicy.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Multi-input policy protocol for actor-critic algorithms.
///
/// Used by A2C, PPO and similar algorithms for Dict observation spaces
/// that may contain multiple input types (images, vectors, etc.).
/// Uses ``CombinedExtractor`` as the default feature extractor.
///
/// - Parameters:
///     - observationSpace: Observation space (Dict).
///     - actionSpace: Action space.
///     - netArch: Network architecture specification.
///     - orthoInit: Whether to use orthogonal initialization.
///     - useSDE: Whether to use State-Dependent Exploration.
///     - logStdInit: Initial value for log standard deviation.
///     - fullStd: Whether to use (features x actions) parameters for std in gSDE.
///     - squashOutput: Whether to squash output using tanh (for gSDE).
///     - shareFeatureExtractor: Whether to share feature extractor between actor and critic.
///     - normalizeImages: Whether to normalize images by dividing by 255.0.
public protocol MultiInputPolicy: ActorCriticPolicy {}

extension MultiInputPolicy {
    /// Creates a CombinedExtractor feature extractor for Dict observations.
    ///
    /// - Returns: A CombinedExtractor configured for the observation space.
    public func makeFeatureExtractor() -> any FeaturesExtractor {
        guard let dict = observationSpace as? Dict else {
            preconditionFailure("MultiInputPolicy requires a Dict observation space")
        }
        return CombinedExtractor(
            observationSpace: dict,
            featuresDim: 256,
            normalizedImage: true
        )
    }
}
