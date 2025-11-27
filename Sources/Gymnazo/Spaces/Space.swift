//
//  Space.swift
//

import Foundation
import MLX

public typealias MaskMLXArray = MLXArray

/// protocol used to define observation and action spaces.
public protocol Space {
    associatedtype T

    var shape: [Int]? { get }
    var dtype: DType? { get }

    /// randomly sample an element of this space using the given RNG key
    /// optionally provide either a mask or a probability distribution.
    func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> T

    /// check if `x` is a valid member of this Space
    func contains(_ x: T) -> Bool

}

extension Space {
    public var shape: [Int]? { nil }
    public var dtype: DType? { nil }

    public func sample(
        key: MLXArray,
        mask: MLXArray? = nil,
        probability: MLXArray? = nil
    ) -> T {
        fatalError("Not implemented")
    }

    /// unmasked/unweighted sample
    public func sample(key: MLXArray) -> T {
        return sample(key: key, mask: nil, probability: nil)
    }

    /// masked sample
    public func sample(key: MLXArray, mask: MLXArray) -> T {
        return sample(key: key, mask: mask, probability: nil)
    }

    /// sample from a probability distribution
    public func sample(key: MLXArray, probability: MLXArray) -> T {
        return sample(key: key, mask: nil, probability: probability)
    }

    func containsAny(_ x: Any) -> Bool {
        if let x = x as? T {
            return contains(x)
        }
        return false
    }

    var isMLXFlattenable: Bool { false }

    func toJsonable(_ sampleN: [T]) -> [Any] {
        return sampleN.map { $0 as Any }
    }

    func fromJsonable(_ sampleN: [Any]) -> [T] {
        return sampleN.compactMap { $0 as? T }
    }
}
