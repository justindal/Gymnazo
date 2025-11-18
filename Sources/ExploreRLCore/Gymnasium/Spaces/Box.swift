//
//  Box.swift
//

import MLX
import MLXRandom

/// continuous space with per-dimension lower and upper bounds.
public struct Box: Space {
    public typealias T = MLXArray

    public let low: MLXArray
    public let high: MLXArray
    public let shape: [Int]?
    public let dtype: DType?

    /// create a Box with scalar bounds broadcast to the provided shape.
    public init(low: Float, high: Float, shape: [Int], dtype: DType = .float32) {
        precondition(high >= low, "Box: high must be >= low")
        self.low = Box.full(shape: shape, value: low, dtype: dtype)
        self.high = Box.full(shape: shape, value: high, dtype: dtype)
        self.shape = shape
        self.dtype = dtype
    }

    /// create a Box with array bounds
    //  arrays must be broadcastable to a common shape.
    public init(low: MLXArray, high: MLXArray, dtype: DType = .float32) {
        // coerce dtype
        let lowC = low.asType(dtype)
        let highC = high.asType(dtype)
        // validate shapes
        let ls = lowC.shape
        let hs = highC.shape
        precondition(ls == hs, "Box: low/high shapes must match")
        self.low = lowC
        self.high = highC
        self.shape = ls
        self.dtype = dtype
    }

    /// returns true if x is within [low, high] elementwise and shape matches.
    public func contains(_ x: MLXArray) -> Bool {
        if let shp = shape {
            let xs = x.shape
            guard shp == xs else { return false }
        }
        let xC = x.asType(dtype ?? .float32)
        let geLow = xC .>= low
        let leHigh = xC .<= high
        let mask = MLX.logicalAnd(geLow.asType(.bool), leHigh.asType(.bool))
        let inRange = MLX.all(mask)
        eval(inRange)
        return (inRange.item() as Bool)
    }

    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> MLXArray {
        // mask and probability not applicable to Box
        let shp = shape ?? low.shape
        let (k, _) = MLXRandom.split(key: key)
        let u01 = MLXRandom.uniform(0 ..< 1, shp, key: k).asType(dtype ?? .float32)
        let span = high - low
        let sample = low + u01 * span
        return sample
    }
}

private extension Box {
    static func full(shape: [Int], value: Float, dtype: DType) -> MLXArray {
        let count = shape.reduce(1, *)
        let flat = MLXArray([Float](repeating: value, count: count))
        let reshaped = flat.reshaped(shape)
        return reshaped.asType(dtype)
    }
}

