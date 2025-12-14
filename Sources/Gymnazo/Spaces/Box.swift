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
    public typealias T = MLXArray

    public let low: MLXArray
    public let high: MLXArray
    public let shape: [Int]?
    public let dtype: DType?
    
    /// Tracks which dimensions are bounded below (not -inf)
    public let boundedBelow: [Bool]
    /// Tracks which dimensions are bounded above (not +inf)
    public let boundedAbove: [Bool]

    /// Create a Box with scalar bounds broadcast to the provided shape.
    public init(low: Float, high: Float, shape: [Int], dtype: DType = .float32) {
        precondition(high >= low, "Box: high must be >= low")
        self.low = Box.full(shape: shape, value: low, dtype: dtype)
        self.high = Box.full(shape: shape, value: high, dtype: dtype)
        self.shape = shape
        self.dtype = dtype
        
        // Scalar bounds
        let count = shape.reduce(1, *)
        self.boundedBelow = [Bool](repeating: !low.isInfinite, count: count)
        self.boundedAbove = [Bool](repeating: !high.isInfinite, count: count)
    }

    /// Create a Box with array bounds.
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
        
        let flatLow = lowC.reshaped([-1]).asArray(Float.self)
        let flatHigh = highC.reshaped([-1]).asArray(Float.self)
        self.boundedBelow = flatLow.map { !$0.isInfinite }
        self.boundedAbove = flatHigh.map { !$0.isInfinite }
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
        let xC = x.asType(dtype ?? .float32)
        
        // For comparison with infinity:
        // x >= -inf is always true for finite x
        // x <= +inf is always true for finite x
        let geLow = xC .>= low
        let leHigh = xC .<= high
        let mask = MLX.logicalAnd(geLow.asType(.bool), leHigh.asType(.bool))
        let inRange = MLX.all(mask)
        eval(inRange)
        return (inRange.item() as Bool)
    }

    /// Sample a random value from the space.
    ///
    /// For unbounded dimensions, samples from a default range:
    /// - Unbounded: samples from [-1e6, 1e6]
    /// - Bounded below only: samples from [low, low + 1e6]
    /// - Bounded above only: samples from [high - 1e6, high]
    ///
    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> MLXArray {
        let shp = shape ?? low.shape
        let (k, _) = MLX.split(key: key)
        let u01 = MLX.uniform(0 ..< 1, shp, key: k).asType(dtype ?? .float32)
        
        let flatLow = low.reshaped([-1]).asArray(Float.self)
        let flatHigh = high.reshaped([-1]).asArray(Float.self)
        
        let defaultRange: Float = 1e6
        var effectiveLow = [Float](repeating: 0, count: flatLow.count)
        var effectiveHigh = [Float](repeating: 0, count: flatHigh.count)
        
        for i in 0..<flatLow.count {
            let bBelow = boundedBelow[i]
            let bAbove = boundedAbove[i]
            
            if bBelow && bAbove {
                // Fully bounded
                effectiveLow[i] = flatLow[i]
                effectiveHigh[i] = flatHigh[i]
            } else if bBelow {
                // Bounded below only
                effectiveLow[i] = flatLow[i]
                effectiveHigh[i] = flatLow[i] + defaultRange
            } else if bAbove {
                // Bounded above only
                effectiveLow[i] = flatHigh[i] - defaultRange
                effectiveHigh[i] = flatHigh[i]
            } else {
                // Unbounded
                effectiveLow[i] = -defaultRange
                effectiveHigh[i] = defaultRange
            }
        }
        
        let effLowArray = MLXArray(effectiveLow).reshaped(shp).asType(dtype ?? .float32)
        let effHighArray = MLXArray(effectiveHigh).reshaped(shp).asType(dtype ?? .float32)
        
        let span = effHighArray - effLowArray
        let sample = effLowArray + u01 * span
        return sample
    }
}

extension Box: MLXSpace {
    public func sampleBatch(key: MLXArray, count: Int) -> MLXArray {
        precondition(count >= 0, "count must be non-negative")
        let elementShape = shape ?? low.shape
        let batchShape = [count] + elementShape
        let u01 = MLX.uniform(0 ..< 1, batchShape, key: key).asType(dtype ?? .float32)

        let flatLow = low.reshaped([-1]).asArray(Float.self)
        let flatHigh = high.reshaped([-1]).asArray(Float.self)

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

        let effLow = MLXArray(effectiveLow).reshaped(elementShape).asType(dtype ?? .float32)
        let effHigh = MLXArray(effectiveHigh).reshaped(elementShape).asType(dtype ?? .float32)

        let span = effHigh - effLow
        return effLow + u01 * span
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
