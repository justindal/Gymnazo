//
//  Model.swift
//  Gymnazo
//

import Foundation
import MLX
import MLXNN

/// Base Model Protocol:
/// Make predictions in response to observations
///
/// For policies, the prediction is an action.
/// For critics, the prediction is the estimated value of the observation.
///
/// - Parameters:
///     - observationSpace: The observation space of the environment.
///     - actionSpace: The action space of the environment.
///     - featuresExtractor: Network to extract features.
///     - normalizeImages: Whether to normalize images or not, dividing by 255.0. True by default.
public protocol Model: Module {
    var observationSpace: any Space<MLXArray> { get }
    var actionSpace: any Space { get }
    var featuresExtractor: (any FeaturesExtractor)? { get }
    var normalizeImages: Bool { get }

    /// Creates a new features extractor instance.
    ///
    /// - Returns: A features extractor configured for the observation space.
    func makeFeatureExtractor() -> any FeaturesExtractor

    /// Preprocesses the observation if needed and extracts features.
    ///
    /// - Parameters:
    ///   - obs: The observation to process.
    ///   - featuresExtractor: The features extractor to use.
    /// - Returns: The extracted features as an MLXArray.
    func extractFeatures(obs: MLXArray, featuresExtractor: any FeaturesExtractor) -> MLXArray
}

extension Model {
    public var normalizeImages: Bool { true }

    public var featuresExtractor: (any FeaturesExtractor)? { nil }

    /// Preprocesses the observation and extracts features using the provided extractor.
    ///
    /// - Parameters:
    ///   - obs: The observation to process.
    ///   - featuresExtractor: The features extractor to use.
    /// - Returns: The extracted features.
    public func extractFeatures(obs: MLXArray, featuresExtractor: any FeaturesExtractor) -> MLXArray
    {
        var preprocessed = obs

        if normalizeImages && obs.dtype == .uint8 {
            preprocessed = obs.asType(.float32) / 255.0
        }

        if let unaryExtractor = featuresExtractor as? any UnaryLayer {
            return unaryExtractor(preprocessed)
        }

        return preprocessed
    }

    /// Extracts features from a Dict observation.
    ///
    /// - Parameters:
    ///   - obs: The Dict observation to process.
    ///   - featuresExtractor: The DictFeaturesExtractor to use.
    /// - Returns: The extracted features.
    public func extractFeatures(
        obs: [String: MLXArray],
        featuresExtractor: any DictFeaturesExtractor
    ) -> MLXArray {
        var pre: [String: MLXArray] = [:]
        for (key, arr) in obs {
            if normalizeImages && arr.dtype == .uint8 {
                pre[key] = arr.asType(.float32) / 255.0
            } else {
                pre[key] = arr
            }
        }
        return featuresExtractor(pre)
    }

    /// Sets the model to training or evaluation mode.
    ///
    /// - Parameter mode: If `true`, set to training mode; otherwise, evaluation mode.
    public func setTrainingMode(_ mode: Bool) {
        self.train(mode)
    }

    /// Saves the model parameters to a file.
    ///
    /// - Parameter url: The file URL to save to.
    /// - Throws: An error if saving fails.
    public func save(to url: URL) throws {
        let flattened = self.parameters().flattened()
        let weights = Dictionary(uniqueKeysWithValues: flattened)
        try MLX.save(arrays: weights, url: url)
    }

    /// Loads model parameters from a file.
    ///
    /// - Parameter url: The file URL to load from.
    /// - Throws: An error if loading fails.
    public func load(from url: URL) throws {
        let loaded = try MLX.loadArrays(url: url)
        try self.update(parameters: ModuleParameters.unflattened(loaded), verify: .noUnusedKeys)
    }
}
