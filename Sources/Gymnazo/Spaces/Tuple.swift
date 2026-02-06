//
//  Tuple.swift
//

import Foundation
import MLX

/// A space representing a tuple of multiple sub-spaces.
/// Samples are concatenated MLXArrays from each sub-space.
public struct Tuple: Space {
    public let spaces: [any Space]

    public init(_ spaces: [any Space]) {
        self.spaces = spaces
    }

    public init(_ spaces: any Space...) {
        self.spaces = spaces
    }

    public var shape: [Int]? {
        nil
    }

    public var dtype: DType? {
        nil
    }

    /// Samples from each sub-space and concatenates the results.
    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> MLXArray {
        let keys = MLX.split(key: key, into: spaces.count)
        var samples: [MLXArray] = []

        for (i, space) in spaces.enumerated() {
            let s = space.sample(key: keys[i], mask: nil, probability: nil)
            samples.append(s.flattened())
        }

        return MLX.concatenated(samples)
    }

    /// Returns `true` if the concatenated array can be split into valid sub-space samples.
    public func contains(_ x: MLXArray) -> Bool {
        var offset = 0
        for space in spaces {
            guard let spaceShape = space.shape else { return false }
            let size = spaceShape.reduce(1, *)
            if offset + size > x.size { return false }
            let slice = x[offset..<(offset + size)].reshaped(spaceShape)
            if !space.contains(slice) { return false }
            offset += size
        }
        return offset == x.size
    }

    public subscript(index: Int) -> any Space {
        return spaces[index]
    }
}
