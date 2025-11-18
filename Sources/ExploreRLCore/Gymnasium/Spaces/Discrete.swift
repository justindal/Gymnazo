//
// Discrete.swift
//

import MLX

public struct Discrete: Space {
    public typealias T = Int
    public let n: Int
    public let start: Int

    public init(n: Int, start: Int = 0) {
        self.n = n
        self.start = start
    }

    public func contains(_ x: Int) -> Bool {
        return (self.start..<(self.start + self.n)).contains(x)
    }

    public func sample(
        key: MLXArray,
        mask: MLXArray? = nil,
        probability: MLXArray? = nil
    ) -> Int {
        if mask != nil && probability != nil {
            fatalError("only one of either mask or probability can be provided")
        }

        if let mask: MLXArray = mask {
            // ensure float32 to avoid GPU float64 issues
            let zero32: MLXArray = MLXArray(0.0 as Float)
            let negInf32: MLXArray = MLXArray(-Float.infinity)
            let logits: MLXArray = MLX.which(mask.asType(.bool), zero32, negInf32)

            eval(logits)

            if logits.max().item() as Float == -Float.infinity {
                return self.start
            }

            let sampled: MLXArray = MLX.categorical(logits, key: key)
            let sampledItem: Int32 = sampled.item() as Int32
            return self.start + Int(sampledItem)
        }

        if let probability: MLXArray = probability {
            let epsilon: MLXArray = MLXArray(1e-9, dtype: .float32)
            let logits: MLXArray = MLX.log(probability + epsilon)

            let sampledIndex: MLXArray = MLX.categorical(logits, key: key)
            let sampledItem: Int32 = sampledIndex.item() as Int32
            return self.start + Int(sampledItem)
        }
        let randomInt: MLXArray = MLX.randInt(low: 0, high: self.n, key: key)
        
        return self.start + Int(randomInt.item() as Int32)
    }
}
