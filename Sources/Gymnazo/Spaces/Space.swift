import Foundation
import MLX

/// protocol used to define observation and action spaces.
public protocol Space<T> {
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

    public func sample(key: MLXArray) -> T {
        sample(key: key, mask: nil, probability: nil)
    }

    public func sample(key: MLXArray, mask: MLXArray) -> T {
        sample(key: key, mask: mask, probability: nil)
    }

    public func sample(key: MLXArray, probability: MLXArray) -> T {
        sample(key: key, mask: nil, probability: probability)
    }
}

/// Attempts to recover a concrete `Box` from a possibly type-erased space.
@inlinable
public func boxSpace<T>(from space: any Space<T>) -> Box? {
    if let box = space as? Box { return box }
    if let erased = space as? AnySpace<T>, let box = erased.base as? Box { return box }
    return nil
}

@inlinable
public func boxSpace(from space: any Space) -> Box? {
    if let box = space as? Box { return box }
    if let erased = space as? AnySpace<MLXArray>, let box = erased.base as? Box { return box }
    return nil
}
