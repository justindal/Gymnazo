//
//  RunningMeanStd.swift
//  Gymnazo
//
//  Generic Welford's algorithm for computing running mean and variance.
//

import Foundation

/// Computes running mean and variance using Welford's online algorithm.
///
/// This is useful for normalizing observations or rewards during training.
///
public final class RunningMeanStd<T: FloatingPoint> {
    public private(set) var mean: T
    public private(set) var varSum: T
    public private(set) var count: T
    
    public init() {
        self.mean = T.zero
        self.varSum = T.zero
        self.count = T.zero
    }
    
    /// Updates the running statistics with a new value.
    public func update(_ x: T) {
        count += 1
        let delta = x - mean
        mean = mean + delta / count
        let delta2 = x - mean
        varSum = varSum + delta * delta2
    }
    
    /// The unbiased sample variance.
    public var variance: T {
        if count < 2 { return T(1) }
        return varSum / (count - 1)
    }
    
    /// Resets the statistics.
    public func reset() {
        mean = T.zero
        varSum = T.zero
        count = T.zero
    }
}

import MLX

/// Computes running mean and variance for MLXArrays using Welford's algorithm.
public final class RunningMeanStdMLX {
    public private(set) var mean: MLXArray
    public private(set) var varSum: MLXArray
    public private(set) var count: Float
    
    public init(shape: [Int]) {
        self.mean = MLX.zeros(shape)
        self.varSum = MLX.zeros(shape)
        self.count = 0
    }
    
    /// Updates the running statistics with a new observation.
    public func update(_ x: MLXArray) {
        count += 1
        let delta = x - mean
        mean = mean + delta / count
        let delta2 = x - mean
        varSum = varSum + delta * delta2
    }
    
    /// The unbiased sample variance.
    public var variance: MLXArray {
        if count < 2 { return MLX.ones(mean.shape) }
        return varSum / (count - 1)
    }
    
    /// The standard deviation.
    public var std: MLXArray {
        MLX.sqrt(variance)
    }
    
    /// Resets the statistics.
    public func reset() {
        mean = MLX.zeros(mean.shape)
        varSum = MLX.zeros(mean.shape)
        count = 0
    }
}
