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
        self.n = n
        self.start = start
    }

    public func contains(_ x: MLXArray) -> Bool {
        let val = Int(x.item(Int32.self))
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
            let zero32: MLXArray = MLXArray(0.0 as Float)
            let negInf32: MLXArray = MLXArray(-Float.infinity)
            let logits: MLXArray = MLX.which(mask.asType(.bool), zero32, negInf32)

            if logits.max().item() as Float == -Float.infinity {
                return MLXArray([Int32(self.start)])
            }

            let sampled: MLXArray = MLX.categorical(logits, key: key)
            let sampledItem: Int32 = sampled.item() as Int32
            return MLXArray([Int32(self.start) + sampledItem])
        }

        if let probability: MLXArray = probability {
            let epsilon: MLXArray = MLXArray(1e-9, dtype: .float32)
            let logits: MLXArray = MLX.log(probability + epsilon)

            let sampledIndex: MLXArray = MLX.categorical(logits, key: key)
            let sampledItem: Int32 = sampledIndex.item() as Int32
            return MLXArray([Int32(self.start) + sampledItem])
        }
        let randomInt: MLXArray = MLX.randInt(low: 0, high: self.n, key: key)
        
        return MLXArray([Int32(self.start) + randomInt.item(Int32.self)])
    }
}
