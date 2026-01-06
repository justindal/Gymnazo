//
//  CNNPolicy.swift
//  Gymnazo
//

import MLX
import MLXNN

/// CNN policy protocol for actor-critic algorithms.
///
/// Used by A2C, PPO and similar algorithms for image observations.
/// Uses ``NatureCNN`` as the default feature extractor.
///
/// - Parameters:
///     - observationSpace: Observation space (Box with 3 dimensions [C, H, W]).
///     - actionSpace: Action space.
///     - netArch: Network architecture specification.
///     - orthoInit: Whether to use orthogonal initialization.
///     - useSDE: Whether to use State-Dependent Exploration.
///     - logStdInit: Initial value for log standard deviation.
///     - fullStd: Whether to use (features x actions) parameters for std in gSDE.
///     - useExpln: Whether to use expln() instead of exp() for positive std.
///     - squashOutput: Whether to squash output using tanh (for gSDE).
///     - shareFeatureExtractor: Whether to share feature extractor between actor and critic.
///     - normalizeImages: Whether to normalize images by dividing by 255.0.
public protocol CNNPolicy: ActorCriticPolicy {}

extension CNNPolicy {
    /// Creates a NatureCNN feature extractor for image observations.
    ///
    /// - Returns: A NatureCNN configured for the observation space.
    public func makeFeatureExtractor() -> any FeaturesExtractor {
        guard let box = observationSpace as? Box else {
            preconditionFailure("CNNPolicy requires a Box observation space")
        }
        return NatureCNN(observationSpace: box, normalizedImage: !normalizeImages)
    }
}
