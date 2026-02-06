//
//  FeaturesExtractor.swift
//  Gymnazo
//
//

import MLX
import MLXNN

/// Protocol for a Features Extractor
///
/// - Parameter featuresDim: Number of features extracted.
public protocol FeaturesExtractor: Module {
    var featuresDim: Int { get }
}

/// Protocol for a Features Extractor that operates on Dict observation spaces.
public protocol DictFeaturesExtractor: FeaturesExtractor {
    func callAsFunction(_ observations: [String: MLXArray]) -> MLXArray
}
