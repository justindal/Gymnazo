//
//  Box.swift
//

import MLX

/// Continuous space with per-dimension lower and upper bounds.
///
/// A Box represents a continuous (possibly unbounded) space in R^n.
/// It supports both bounded and unbounded dimensions using infinity values.
///
/// ## Example
/// ```swift
/// // Bounded box
/// let bounded = Box(low: -1, high: 1, shape: [4])
///
/// let unbounded = Box(
///     low: MLXArray([-.infinity, -.infinity, 0.0] as [Float32]),
///     high: MLXArray([.infinity, .infinity, 1.0] as [Float32])
/// )
/// ```
public struct Box: Space {
    public let low: MLXArray
    public let high: MLXArray
    public let shape: [Int]?
    public let dtype: DType?

    /// Tracks which dimensions are bounded below (not -inf)
    public let boundedBelow: [Bool]
    /// Tracks which dimensions are bounded above (not +inf)
    public let boundedAbove: [Bool]

    private let effectiveLow: MLXArray
    private let effectiveHigh: MLXArray

    /// Create a Box with scalar bounds broadcast to the provided shape.
    public init(low: Float, high: Float, shape: [Int], dtype: DType = .float32) {
        precondition(high >= low, "Box: high must be >= low")
        self.low = Box.full(shape: shape, value: low, dtype: dtype)
        self.high = Box.full(shape: shape, value: high, dtype: dtype)
        self.shape = shape
        self.dtype = dtype

        let count = shape.reduce(1, *)
        self.boundedBelow = [Bool](repeating: low > -.infinity, count: count)
        self.boundedAbove = [Bool](repeating: high < .infinity, count: count)

        let flatLow = [Float](repeating: low, count: count)
        let flatHigh = [Float](repeating: high, count: count)
        let (effLow, effHigh) = Box.computeEffectiveBounds(
            flatLow: flatLow,
            flatHigh: flatHigh,
            boundedBelow: self.boundedBelow,
            boundedAbove: self.boundedAbove
        )
        self.effectiveLow = MLXArray(effLow).reshaped(shape).asType(dtype)
        self.effectiveHigh = MLXArray(effHigh).reshaped(shape).asType(dtype)
    }

    /// Create a Box with array bounds.
    public init(low: MLXArray, high: MLXArray, dtype: DType = .float32) {
        let lowC = low.asType(dtype)
        let highC = high.asType(dtype)
        let ls = lowC.shape
        let hs = highC.shape
        precondition(ls == hs, "Box: low/high shapes must match")
        self.low = lowC
        self.high = highC
        self.shape = ls
        self.dtype = dtype

        let flatLow = lowC.reshaped([-1]).asArray(Float.self)
        let flatHigh = highC.reshaped([-1]).asArray(Float.self)
        precondition(
            zip(flatLow, flatHigh).allSatisfy {
                !$0.isNaN && !$1.isNaN && $0 <= $1
            },
            "Box: every low bound must be <= its high bound and neither may be NaN"
        )
        self.boundedBelow = flatLow.map { $0 > -.infinity }
        self.boundedAbove = flatHigh.map { $0 < .infinity }

        let (effLow, effHigh) = Box.computeEffectiveBounds(
            flatLow: flatLow,
            flatHigh: flatHigh,
            boundedBelow: self.boundedBelow,
            boundedAbove: self.boundedAbove
        )
        self.effectiveLow = MLXArray(effLow).reshaped(ls).asType(dtype)
        self.effectiveHigh = MLXArray(effHigh).reshaped(ls).asType(dtype)
    }

    /// Returns whether the space is bounded in all dimensions.
    ///
    /// - Parameter manner: How to check boundedness:
    ///   - `"below"`: Check if bounded below (low > -inf)
    ///   - `"above"`: Check if bounded above (high < +inf)
    ///   - `"both"`: Check if bounded in both directions (default)
    /// - Returns: `true` if the space is bounded in the specified manner
    public func isBounded(manner: String = "both") -> Bool {
        switch manner {
        case "below":
            return boundedBelow.allSatisfy { $0 }
        case "above":
            return boundedAbove.allSatisfy { $0 }
        case "both":
            return boundedBelow.allSatisfy { $0 } && boundedAbove.allSatisfy { $0 }
        default:
            preconditionFailure("manner must be 'below', 'above', or 'both'")
        }
    }

    /// Returns true if x is within [low, high] elementwise and shape matches.
    /// Properly handles infinite bounds - any finite value satisfies an infinite bound.
    public func contains(_ x: MLXArray) -> Bool {
        if let shp = shape {
            let xs = x.shape
            guard shp == xs else { return false }
        }
        if dtype?.isInteger == true, !x.dtype.isInteger {
            return false
        }
        let xC = x.asType(dtype ?? .float32)

        let geLow = xC .>= low
        let leHigh = xC .<= high
        let mask = MLX.logicalAnd(geLow.asType(.bool), leHigh.asType(.bool))
        let inRange = MLX.all(mask)
        eval(inRange)
        return inRange.scalarValue(Bool.self)
    }

    /// Sample a random value from the space.
    ///
    /// Fully bounded dimensions use a uniform distribution. Lower-bounded and
    /// upper-bounded dimensions use shifted exponential distributions, while
    /// fully unbounded dimensions use a standard normal distribution.
    ///
    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> MLXArray {
        precondition(mask == nil && probability == nil, "Box does not support mask or probability")
        let shp = shape ?? low.shape
        return sample(shape: shp, key: key)
    }
}

extension Box: TensorSpace {
    public func sampleBatch(key: MLXArray, count: Int) -> MLXArray {
        precondition(count >= 0, "count must be non-negative")
        let elementShape = shape ?? low.shape
        let batchShape = [count] + elementShape
        return sample(shape: batchShape, key: key)
    }
}

extension Box {
    private func sample(shape sampleShape: [Int], key: MLXArray) -> MLXArray {
        let keys = MLX.split(key: key, into: 2)
        let uniform = MLX.uniform(0..<1, sampleShape, key: keys[0]).asType(.float32)
        let exponential = -MLX.log(MLX.maximum(1 - uniform, 1e-7))
        let normal = MLXRandom.normal(sampleShape, key: keys[1]).asType(.float32)
        let belowShape =
            sampleShape.count == (shape ?? low.shape).count
            ? (shape ?? low.shape)
            : [1] + (shape ?? low.shape)
        let below = MLXArray(boundedBelow).reshaped(belowShape)
        let above = MLXArray(boundedAbove).reshaped(belowShape)
        let bounded = MLX.logicalAnd(below, above)
        let lowerOnly = MLX.logicalAnd(below, MLX.logicalNot(above))
        let upperOnly = MLX.logicalAnd(MLX.logicalNot(below), above)
        let uniformSample = effectiveLow + uniform * (effectiveHigh - effectiveLow)
        let sample = MLX.which(
            bounded,
            uniformSample,
            MLX.which(
                lowerOnly,
                low + exponential,
                MLX.which(upperOnly, high - exponential, normal)
            )
        )
        return sample.asType(dtype ?? .float32)
    }

    fileprivate static func full(shape: [Int], value: Float, dtype: DType) -> MLXArray {
        let count = shape.reduce(1, *)
        let flat = MLXArray([Float](repeating: value, count: count))
        let reshaped = flat.reshaped(shape)
        return reshaped.asType(dtype)
    }

    /// Computes effective bounds for sampling, handling unbounded dimensions.
    fileprivate static func computeEffectiveBounds(
        flatLow: [Float],
        flatHigh: [Float],
        boundedBelow: [Bool],
        boundedAbove: [Bool]
    ) -> ([Float], [Float]) {
        let defaultRange: Float = 1e6
        var effectiveLow = [Float](repeating: 0, count: flatLow.count)
        var effectiveHigh = [Float](repeating: 0, count: flatHigh.count)

        for i in 0..<flatLow.count {
            let bBelow = boundedBelow[i]
            let bAbove = boundedAbove[i]

            if bBelow && bAbove {
                effectiveLow[i] = flatLow[i]
                effectiveHigh[i] = flatHigh[i]
            } else if bBelow {
                effectiveLow[i] = flatLow[i]
                effectiveHigh[i] = flatLow[i] + defaultRange
            } else if bAbove {
                effectiveLow[i] = flatHigh[i] - defaultRange
                effectiveHigh[i] = flatHigh[i]
            } else {
                effectiveLow[i] = -defaultRange
                effectiveHigh[i] = defaultRange
            }
        }

        return (effectiveLow, effectiveHigh)
    }
}
