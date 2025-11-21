//
//  Dict.swift
//

import Foundation
import MLX
import MLXRandom

public struct Dict: Space {
    public typealias T = [String: Any]
    
    public let spaces: [String: any Space]
    
    public init(_ spaces: [String: any Space]) {
        self.spaces = spaces
    }
    
    public var shape: [Int]? {
        return nil
    }
    
    public var dtype: DType? {
        return nil
    }
    
    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> [String: Any] {
        let keys = MLXRandom.split(key: key, num: spaces.count)
        var sample: [String: Any] = [:]
        
        let sortedKeys = spaces.keys.sorted()
        
        for (i, k) in sortedKeys.enumerated() {
            let space = spaces[k]!
            sample[k] = space.sample(key: keys[i], mask: nil, probability: nil)
        }
        
        return sample
    }
    
    public func contains(_ x: [String: Any]) -> Bool {
        for (k, space) in spaces {
            guard let value = x[k] else { return false }
            if !space.containsAny(value) {
                return false
            }
        }
        return true
    }
    
    public subscript(key: String) -> (any Space)? {
        return spaces[key]
    }
}
