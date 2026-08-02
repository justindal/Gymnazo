//
// Discrete.swift
//

import MLX

public struct Discrete: Space {
    public let n: Int
    public let start: Int
    public var shape: [Int]? { [1] }
    public var dtype: DType? { .int32 }

    public init(n: Int, start: Int = 0) {
        precondition(n > 0, "Discrete requires n > 0")
        precondition(!start.addingReportingOverflow(n).overflow, "Discrete bounds overflow Int")
        self.n = n
        self.start = start
    }

    public func contains(_ x: MLXArray) -> Bool {
        guard x.size == 1, x.dtype.isInteger else { return false }
        let value = x.asType(.int64).singletonValue(Int64.self)
        guard let val = Int(exactly: value) else { return false }
        return (self.start..<(self.start + self.n)).contains(val)
    }

    public func sample(
        key: MLXArray,
        mask: MLXArray? = nil,
        probability: MLXArray? = nil
    ) -> MLXArray {
        if mask != nil && probability != nil {
            fatalError("only one of either mask or probability can be provided")
        }

        if let mask: MLXArray = mask {
            precondition(mask.shape == [n], "Discrete mask must have shape [n]")
            let validMask = MLX.logicalOr(mask .== 0, mask .== 1)
            precondition(validMask.all().item(Bool.self), "Discrete mask values must be 0 or 1")
            let zero32: MLXArray = MLXArray(0.0 as Float)
            let negInf32: MLXArray = MLXArray(-Float.infinity)
            let logits: MLXArray = MLX.which(mask.asType(.bool), zero32, negInf32)

            if logits.max().scalarValue(Float.self) == -Float.infinity {
                return MLXArray([Int32(self.start)])
            }

            let sampled: MLXArray = MLX.categorical(logits, key: key)
            let sampledItem = sampled.scalarValue(Int32.self)
            return MLXArray([Int32(self.start) + sampledItem])
        }

        if let probability: MLXArray = probability {
            precondition(probability.shape == [n], "Discrete probability must have shape [n]")
            precondition(
                (probability .>= 0).all().item(Bool.self), "Probabilities must be non-negative")
            let total = probability.asType(.float32).sum().item(Float.self)
            precondition(abs(total - 1) <= 0.000_001, "Discrete probabilities must sum to one")
            let epsilon: MLXArray = MLXArray(1e-9, dtype: .float32)
            let logits: MLXArray = MLX.log(probability + epsilon)

            let sampledIndex: MLXArray = MLX.categorical(logits, key: key)
            let sampledItem = sampledIndex.scalarValue(Int32.self)
            return MLXArray([Int32(self.start) + sampledItem])
        }
        let randomInt: MLXArray = MLX.randInt(low: 0, high: self.n, key: key)

        return MLXArray([Int32(self.start) + randomInt.scalarValue(Int32.self)])
    }
}
