import MLX

/// A type-erased wrapper around any ``MLXSpace``.
public struct AnySpace: MLXSpace {
    public typealias T = MLXArray

    public let shape: [Int]?
    public let dtype: DType?

    private let sampleFn: (MLXArray, MLXArray?, MLXArray?) -> MLXArray
    private let containsFn: (MLXArray) -> Bool
    private let sampleBatchFn: (MLXArray, Int) -> MLXArray

    public init<S: MLXSpace>(_ space: S) {
        self.shape = space.shape
        self.dtype = space.dtype
        self.sampleFn = { key, mask, probability in
            space.sample(key: key, mask: mask, probability: probability)
        }
        self.containsFn = { x in
            space.contains(x)
        }
        self.sampleBatchFn = { key, count in
            space.sampleBatch(key: key, count: count)
        }
    }

    public init(_ space: any MLXSpace) {
        self.shape = space.shape
        self.dtype = space.dtype
        self.sampleFn = { key, mask, probability in
            space.sample(key: key, mask: mask, probability: probability)
        }
        self.containsFn = { x in
            space.contains(x)
        }
        self.sampleBatchFn = { key, count in
            space.sampleBatch(key: key, count: count)
        }
    }

    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> MLXArray {
        sampleFn(key, mask, probability)
    }

    public func contains(_ x: MLXArray) -> Bool {
        containsFn(x)
    }

    public func sampleBatch(key: MLXArray, count: Int) -> MLXArray {
        sampleBatchFn(key, count)
    }
}

