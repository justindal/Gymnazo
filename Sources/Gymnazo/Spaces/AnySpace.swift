import MLX

/// A type-erased wrapper around any Space.
public struct AnySpace<Element>: Space {
    public typealias T = Element

    public let base: any Space<Element>

    public var shape: [Int]? { base.shape }
    public var dtype: DType? { base.dtype }

    public init(_ space: any Space<Element>) {
        self.base = space
    }

    public init<S: Space>(_ space: S) where S.T == Element {
        self.base = space
    }

    public func sample(
        key: MLXArray,
        mask: MLXArray?,
        probability: MLXArray?
    ) -> Element {
        base.sample(key: key, mask: mask, probability: probability)
    }

    public func contains(_ x: Element) -> Bool {
        base.contains(x)
    }
}
