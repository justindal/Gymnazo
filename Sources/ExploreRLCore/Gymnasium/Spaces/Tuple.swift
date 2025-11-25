//
//  Tuple.swift
//

import Foundation
import MLX

public struct Tuple: Space {
    public typealias T = [Any]
    
    public let spaces: [any Space]
    
    public init(_ spaces: [any Space]) {
        self.spaces = spaces
    }
    
    public init(_ spaces: any Space...) {
        self.spaces = spaces
    }
    
    public var shape: [Int]? {
        return nil
    }
    
    public var dtype: DType? {
        return nil
    }
    
    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> [Any] {
        let keys = MLX.split(key: key, into: spaces.count)
        var sample: [Any] = []
        
        for (i, space) in spaces.enumerated() {
            sample.append(space.sample(key: keys[i], mask: nil, probability: nil))
        }
        
        return sample
    }
    
    public func contains(_ x: [Any]) -> Bool {
        guard x.count == spaces.count else { return false }
        
        for (i, space) in spaces.enumerated() {
            if !space.containsAny(x[i]) {
                return false
            }
        }
        return true
    }
    
    public subscript(index: Int) -> any Space {
        return spaces[index]
    }
}
