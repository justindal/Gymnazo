//
//  MultiDiscrete.swift
//

import Foundation
import MLX

public struct MultiDiscrete: Space {
    public let nvec: MLXArray
    public let start: MLXArray
    public let shape: [Int]?
    public let dtype: DType? = .int32
    private let counts: [Int]
    private let starts: [Int]

    public init(_ nvec: [Int], start: [Int]? = nil) {
        let starts = start ?? [Int](repeating: 0, count: nvec.count)
        precondition(!nvec.isEmpty, "MultiDiscrete requires at least one dimension")
        precondition(starts.count == nvec.count, "MultiDiscrete start must match nvec")
        precondition(nvec.allSatisfy { $0 > 0 }, "MultiDiscrete counts must be positive")
        precondition(
            zip(starts, nvec).allSatisfy { !$0.addingReportingOverflow($1).overflow },
            "MultiDiscrete bounds overflow Int"
        )
        self.counts = nvec
        self.starts = starts
        self.nvec = MLXArray(nvec).asType(.int32)
        self.start = MLXArray(starts).asType(.int32)
        self.shape = [nvec.count]
    }

    public init(_ nvec: MLXArray, start: MLXArray? = nil) {
        let counts = nvec.asType(.int32).asArray(Int32.self).map(Int.init)
        let starts = (start ?? MLX.zeros(nvec.shape, dtype: .int32))
            .asType(.int32).asArray(Int32.self).map(Int.init)
        precondition(nvec.shape == (start?.shape ?? nvec.shape), "MultiDiscrete bounds must match")
        precondition(!counts.isEmpty, "MultiDiscrete requires at least one dimension")
        precondition(counts.allSatisfy { $0 > 0 }, "MultiDiscrete counts must be positive")
        precondition(
            zip(starts, counts).allSatisfy { !$0.addingReportingOverflow($1).overflow },
            "MultiDiscrete bounds overflow Int"
        )
        self.counts = counts
        self.starts = starts
        self.nvec = nvec.asType(.int32)
        self.start = (start ?? MLX.zeros(nvec.shape, dtype: .int32)).asType(.int32)
        self.shape = nvec.shape
    }

    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> MLXArray {
        precondition(mask == nil || probability == nil, "Provide either mask or probability")
        guard mask != nil || probability != nil else {
            return MLX.randInt(
                low: start,
                high: start + nvec,
                shape,
                key: key
            ).asType(.int32)
        }
        let width = counts.max() ?? 0
        let source = mask ?? probability!
        precondition(
            source.shape == [counts.count, width],
            "MultiDiscrete masks and probabilities use padded shape [dimensions, max(nvec)]"
        )
        let keys = MLX.split(key: key, into: counts.count)
        var samples: [MLXArray] = []
        samples.reserveCapacity(counts.count)
        for index in counts.indices {
            let count = counts[index]
            let row = source[index, 0..<count]
            let logits: MLXArray
            if mask != nil {
                let valid = MLX.logicalOr(row .== 0, row .== 1)
                precondition(
                    valid.all().item(Bool.self), "MultiDiscrete mask values must be 0 or 1")
                if row.asType(.int32).sum().item(Int32.self) == 0 {
                    samples.append(MLXArray(Int32(starts[index])))
                    continue
                }
                logits = MLX.which(
                    row.asType(.bool),
                    MLXArray(Float(0)),
                    MLXArray(-Float.infinity)
                )
            } else {
                precondition(
                    (row .>= 0).all().item(Bool.self), "Probabilities must be non-negative")
                let total = row.asType(.float32).sum().item(Float.self)
                precondition(abs(total - 1) <= 0.000_001, "Each probability row must sum to one")
                logits = MLX.log(row.asType(.float32) + 1e-9)
            }
            samples.append(
                MLX.categorical(logits, key: keys[index]).asType(.int32)
                    + Int32(starts[index])
            )
        }
        return MLX.stacked(samples).reshaped(shape!)
    }

    public func contains(_ x: MLXArray) -> Bool {
        if x.shape != self.shape || !x.dtype.isInteger { return false }

        let lower = (x .>= start).all().scalarValue(Bool.self)
        let upper = (x .< start + self.nvec).all().scalarValue(Bool.self)
        return lower && upper
    }
}

extension MultiDiscrete: TensorSpace {
    public func sampleBatch(key: MLXArray, count: Int) -> MLXArray {
        precondition(count >= 0, "count must be non-negative")
        guard let elementShape = self.shape else {
            fatalError("MultiDiscrete requires a defined shape")
        }
        let batchShape = [count] + elementShape
        return MLX.randInt(
            low: start,
            high: start + nvec,
            batchShape,
            key: key
        ).asType(.int32)
    }
}
