//
//  Dict.swift
//

import Foundation
import MLX

/// A space representing a dictionary of named sub-spaces.
/// Samples are concatenated MLXArrays from each sub-space (sorted by key).
public struct Dict: Space {
    public let spaces: [String: any Space]
    
    public init(_ spaces: [String: any Space]) {
        self.spaces = spaces
    }
    
    public var shape: [Int]? {
        nil
    }
    
    public var dtype: DType? {
        nil
    }
    
    /// Samples from each sub-space and concatenates the results (sorted by key).
    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> MLXArray {
        let sortedKeys = spaces.keys.sorted()
        let keys = MLX.split(key: key, into: sortedKeys.count)
        var samples: [MLXArray] = []
        
        for (i, k) in sortedKeys.enumerated() {
            let space = spaces[k]!
            let s = space.sample(key: keys[i], mask: nil, probability: nil)
            samples.append(s.flattened())
        }
        
        return MLX.concatenated(samples)
    }
    
    /// Returns `true` if the concatenated array can be split into valid sub-space samples.
    public func contains(_ x: MLXArray) -> Bool {
        let sortedKeys = spaces.keys.sorted()
        var offset = 0
        for k in sortedKeys {
            let space = spaces[k]!
            guard let spaceShape = space.shape else { return false }
            let size = spaceShape.reduce(1, *)
            if offset + size > x.size { return false }
            let slice = x[offset..<(offset + size)].reshaped(spaceShape)
            if !space.contains(slice) { return false }
            offset += size
        }
        return offset == x.size
    }
    
    public subscript(key: String) -> (any Space)? {
        return spaces[key]
    }
}
