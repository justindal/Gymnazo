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
public protocol FeaturesExtractor: Module, UnaryLayer {
    var featuresDim: Int { get }
}

// Protocol for Features Extractor for Dict Space
public protocol DictFeaturesExtractor: Module {
    var featuresDim: Int { get }
    func callAsFunction(_ observations: [String: MLXArray]) -> MLXArray
}
