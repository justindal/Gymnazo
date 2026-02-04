//
//  Schedules.swift
//  Gymnazo
//

import Foundation

/// Linear learning rate schedule that decays from initial to final value.
public struct LinearSchedule: LearningRateSchedule {
    public let initialValue: Double
    public let finalValue: Double

    /// Creates a linear learning rate schedule.
    ///
    /// - Parameters:
    ///   - initialValue: Starting learning rate.
    ///   - finalValue: Ending learning rate.
    public init(initialValue: Double, finalValue: Double = 0) {
        self.initialValue = initialValue
        self.finalValue = finalValue
    }

    public func value(at progressRemaining: Double) -> Double {
        finalValue + progressRemaining * (initialValue - finalValue)
    }
}

/// Exponential learning rate schedule.
public struct ExponentialSchedule: LearningRateSchedule {
    public let initialValue: Double
    public let decayRate: Double

    /// Creates an exponential learning rate schedule.
    ///
    /// - Parameters:
    ///   - initialValue: Starting learning rate.
    ///   - decayRate: Decay rate per progress unit.
    public init(initialValue: Double, decayRate: Double = 0.99) {
        self.initialValue = initialValue
        self.decayRate = decayRate
    }

    public func value(at progressRemaining: Double) -> Double {
        let progressDone = 1.0 - progressRemaining
        return initialValue * pow(decayRate, progressDone * 100)
    }
}

/// Step learning rate schedule that reduces at specified milestones.
public struct StepSchedule: LearningRateSchedule {
    public let initialValue: Double
    public let milestones: [Double]
    public let gamma: Double

    /// Creates a step learning rate schedule.
    ///
    /// - Parameters:
    ///   - initialValue: Starting learning rate.
    ///   - milestones: Progress values (0-1) at which to reduce LR.
    ///   - gamma: Multiplicative factor for each step.
    public init(initialValue: Double, milestones: [Double], gamma: Double = 0.1) {
        self.initialValue = initialValue
        self.milestones = milestones.sorted(by: >)
        self.gamma = gamma
    }

    public func value(at progressRemaining: Double) -> Double {
        let progressDone = 1.0 - progressRemaining
        var lr = initialValue
        for milestone in milestones {
            if progressDone >= milestone {
                lr *= gamma
            }
        }
        return lr
    }
}

/// Cosine annealing learning rate schedule.
public struct CosineAnnealingSchedule: LearningRateSchedule {
    public let initialValue: Double
    public let minValue: Double

    /// Creates a cosine annealing learning rate schedule.
    ///
    /// - Parameters:
    ///   - initialValue: Starting learning rate.
    ///   - minValue: Minimum learning rate.
    public init(initialValue: Double, minValue: Double = 0) {
        self.initialValue = initialValue
        self.minValue = minValue
    }

    public func value(at progressRemaining: Double) -> Double {
        let progressDone = 1.0 - progressRemaining
        return minValue + 0.5 * (initialValue - minValue) * (1 + cos(Double.pi * progressDone))
    }
}

/// Warmup learning rate schedule that linearly increases then follows another schedule.
public struct WarmupSchedule: LearningRateSchedule {
    public let baseSchedule: any LearningRateSchedule
    public let warmupFraction: Double
    public let warmupInitialValue: Double

    /// Creates a warmup learning rate schedule.
    ///
    /// - Parameters:
    ///   - baseSchedule: Schedule to follow after warmup.
    ///   - warmupFraction: Fraction of training for warmup (0-1).
    ///   - warmupInitialValue: Initial value during warmup.
    public init(
        baseSchedule: any LearningRateSchedule,
        warmupFraction: Double = 0.05,
        warmupInitialValue: Double = 0
    ) {
        self.baseSchedule = baseSchedule
        self.warmupFraction = warmupFraction
        self.warmupInitialValue = warmupInitialValue
    }

    public func value(at progressRemaining: Double) -> Double {
        let progressDone = 1.0 - progressRemaining

        if progressDone < warmupFraction {
            let warmupProgress = progressDone / warmupFraction
            let targetValue = baseSchedule.value(at: 1.0 - warmupFraction)
            return warmupInitialValue + warmupProgress * (targetValue - warmupInitialValue)
        }

        let adjustedProgress = (progressDone - warmupFraction) / (1.0 - warmupFraction)
        return baseSchedule.value(at: 1.0 - adjustedProgress)
    }
}

/// Callable wrapper for learning rate schedules.
public struct ScheduleFunction: LearningRateSchedule {
    private let function: (Double) -> Double

    /// Creates a schedule from a custom function.
    ///
    /// - Parameter function: Function mapping progress remaining to learning rate.
    public init(_ function: @escaping (Double) -> Double) {
        self.function = function
    }

    public func value(at progressRemaining: Double) -> Double {
        function(progressRemaining)
    }

    public func callAsFunction(_ progressRemaining: Double) -> Double {
        function(progressRemaining)
    }
}
