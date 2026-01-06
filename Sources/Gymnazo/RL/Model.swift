//
//  Model.swift
//  Gymnazo
//

import MLX
import MLXNN
import MLXOptimizers

/// Base Model Protocol:
/// Make predictions in response to observations
///
/// For policies, the prediction is an action.
/// For critics, the prediction is the estimated value of the observation.
///
/// - Parameters:
///     - observationSpace: The observation space of the environment.
///     - actionSpace: The action space of the environment.
///     - featuresExtractorClass: Features extractor to use.
///     - featuresExtractorKwargs: Keyword arguments to pass to the features extractor.
///     - featuresExtractor: Network to extract features.
///     - normalizeImages: Whether to normalize images or not, dividing by 255.0. True by default.
///     - optimizerClass: The optimizer to use. `MLXOptimizers.Adam` is default.
///     - optimizerKwargs: Keyword arguements to pass to the optimizer.
///
protocol Model: Module {
    var observationSpace: Space { get }
    var actionSpace: Space { get }
    
    
}
