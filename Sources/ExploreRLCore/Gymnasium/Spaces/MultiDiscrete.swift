//
//  MultiDiscrete.swift
//

import Foundation
import MLX
import MLXRandom

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
        let rand = MLXRandom.uniform(low: 0, high: 1, shape: self.shape!, key: key)
        return (rand * self.nvec).astype(.int32)
    }
    
    public func contains(_ x: MLXArray) -> Bool {
        if x.shape != self.shape { return false }
        
        let lower = (x .>= 0).all().item(Bool.self)
        let upper = (x .< self.nvec).all().item(Bool.self)
        return lower && upper
    }
}
