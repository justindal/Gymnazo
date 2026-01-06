//
//  FlattenExtractor.swift
//  Gymnazo
//
//

import MLX
import MLXNN

public protocol FlattenableSpace {
    var shape: [Int]? { get }
}

public class FlattenExtractor: Module, UnaryLayer, FeaturesExtractor {
    public let featuresDim: Int
    
    public init(featuresDim: Int) {
        precondition(featuresDim > 0)
        self.featuresDim = featuresDim
        super.init()
    }
    
    public convenience init<S: FlattenableSpace>(_ observationSpace: S) {
        guard let shape = observationSpace.shape, !shape.isEmpty else {
            preconditionFailure("")
        }
        let dim = shape.reduce(1, *)
        precondition(dim > 0)
        self.init(featuresDim: dim)
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        
        if shape.count <= 3 {
            return x
        }
        
        let batch = shape[0]
        let rest = shape.dropFirst().reduce(1, *)
        return x.reshaped([batch, rest])
    }
    
    
}
