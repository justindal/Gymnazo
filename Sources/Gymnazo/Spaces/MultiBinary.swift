import MLX

public struct MultiBinary: Space {
    public typealias T = MLXArray

    public let shape: [Int]?
    public let dtype: DType? = .int32

    public init(n: Int) {
        precondition(n >= 0, "n must be non-negative")
        self.shape = [n]
    }

    public init(shape: [Int]) {
        precondition(shape.allSatisfy { $0 >= 0 }, "shape must be non-negative")
        self.shape = shape
    }

    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> MLXArray {
        guard let shape else {
            fatalError("MultiBinary requires a defined shape")
        }
        let u01 = MLX.uniform(low: 0, high: 1, shape, key: key).asType(.float32)
        return (u01 .>= 0.5).asType(.int32)
    }

    public func contains(_ x: MLXArray) -> Bool {
        if let shape, x.shape != shape { return false }
        let xI = x.asType(.int32)
        let ge0 = (xI .>= 0).all().item(Bool.self)
        let le1 = (xI .<= 1).all().item(Bool.self)
        return ge0 && le1
    }
}

extension MultiBinary: MLXSpace {
    public func sampleBatch(key: MLXArray, count: Int) -> MLXArray {
        precondition(count >= 0, "count must be non-negative")
        guard let elementShape = shape else {
            fatalError("MultiBinary requires a defined shape")
        }
        let batchShape = [count] + elementShape
        let u01 = MLX.uniform(low: 0, high: 1, batchShape, key: key).asType(.float32)
        return (u01 .>= 0.5).asType(.int32)
    }
}

