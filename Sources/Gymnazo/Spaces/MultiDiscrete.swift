//
//  MultiDiscrete.swift
//

import Foundation
import MLX

public struct MultiDiscrete: Space {
    public typealias T = MLXArray
    
    public let nvec: MLXArray
    public let shape: [Int]?
    public let dtype: DType? = .int32
    
    public init(_ nvec: [Int]) {
        self.nvec = MLXArray(nvec)
        self.shape = [nvec.count]
    }
    
    public init(_ nvec: MLXArray) {
        self.nvec = nvec
        self.shape = nvec.shape
    }
    
    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> MLXArray {
        let rand = MLX.uniform(low: 0, high: 1, self.shape!, key: key)
        return (rand * self.nvec).asType(.int32)
    }
    
    public func contains(_ x: MLXArray) -> Bool {
        if x.shape != self.shape { return false }
        
        let lower = (x .>= 0).all().item(Bool.self)
        let upper = (x .< self.nvec).all().item(Bool.self)
        return lower && upper
    }
}

extension MultiDiscrete: TensorSpace {
    public func sampleBatch(key: MLXArray, count: Int) -> MLXArray {
        precondition(count >= 0, "count must be non-negative")
        guard let elementShape = self.shape else {
            fatalError("MultiDiscrete requires a defined shape")
        }
        let batchShape = [count] + elementShape
        let rand = MLX.uniform(low: 0, high: 1, batchShape, key: key)
        return (rand * self.nvec).asType(.int32)
    }
}
