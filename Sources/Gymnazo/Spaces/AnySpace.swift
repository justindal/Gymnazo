import MLX

public struct AnySpace: Space {
    public let base: any Space

    public var shape: [Int]? { base.shape }
    public var dtype: DType? { base.dtype }

    public init(_ space: any Space) {
        self.base = space
    }

    public func sample(
        key: MLXArray,
        mask: MLXArray?,
        probability: MLXArray?
    ) -> MLXArray {
        base.sample(key: key, mask: mask, probability: probability)
    }

    public func contains(_ x: MLXArray) -> Bool {
        base.contains(x)
    }
}
