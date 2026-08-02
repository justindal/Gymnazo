import Foundation
import MLX

public protocol Space {
    var shape: [Int]? { get }
    var dtype: DType? { get }

    func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> MLXArray

    func contains(_ x: MLXArray) -> Bool
}

extension Space {
    public var shape: [Int]? { nil }
    public var dtype: DType? { nil }

    public func sample(
        key: MLXArray,
        mask: MLXArray? = nil,
        probability: MLXArray? = nil
    ) -> MLXArray {
        fatalError("Not implemented")
    }

    public func sample(key: MLXArray) -> MLXArray {
        sample(key: key, mask: nil, probability: nil)
    }

    public func sample(key: MLXArray, mask: MLXArray) -> MLXArray {
        sample(key: key, mask: mask, probability: nil)
    }

    public func sample(key: MLXArray, probability: MLXArray) -> MLXArray {
        sample(key: key, mask: nil, probability: probability)
    }
}

@inlinable
public func boxSpace(from space: any Space) -> Box? {
    if let box = space as? Box { return box }
    if let erased = space as? AnySpace, let box = erased.base as? Box { return box }
    return nil
}
