//
//  StopTrainingCallbacks.swift
//  Gymnazo
//

import Foundation

/// Callback that stops training when mean reward exceeds a threshold.
public final class StopTrainingOnRewardThreshold: Callback {
    public let rewardThreshold: Double
    public let verbose: Int

    private let rewardProvider: () -> Double?
    private var stopped: Bool = false

    /// Whether training was stopped by this callback.
    public var hasStopped: Bool { stopped }

    /// Creates a stop training callback based on reward threshold.
    ///
    /// - Parameters:
    ///   - rewardThreshold: Stop when mean reward exceeds this value.
    ///   - rewardProvider: Closure that returns the current mean reward.
    ///   - verbose: Verbosity level.
    public init(
        rewardThreshold: Double,
        rewardProvider: @escaping () -> Double?,
        verbose: Int = 1
    ) {
        self.rewardThreshold = rewardThreshold
        self.rewardProvider = rewardProvider
        self.verbose = verbose
    }

    public func onStep(locals: CallbackLocals) -> Bool {
        guard let meanReward = rewardProvider() else { return true }

        if meanReward >= rewardThreshold {
            stopped = true
            if verbose >= 1 {
                print(
                    "Stopping training: mean reward \(String(format: "%.2f", meanReward)) >= threshold \(String(format: "%.2f", rewardThreshold))"
                )
            }
            return false
        }

        return true
    }
}

/// Callback that stops training if no improvement is seen for a number of evaluations.
public final class StopTrainingOnNoImprovement: Callback {
    public let maxNoImprovementEvals: Int
    public let minEvals: Int
    public let verbose: Int

    private let valueProvider: () -> Double?
    private var bestValue: Double = -.infinity
    private var noImprovementCount: Int = 0
    private var evalCount: Int = 0
    private var stopped: Bool = false
    private var lastValue: Double?

    /// Whether training was stopped by this callback.
    public var hasStopped: Bool { stopped }

    /// Current number of evaluations without improvement.
    public var currentNoImprovementCount: Int { noImprovementCount }

    /// Creates a stop training callback based on lack of improvement.
    ///
    /// - Parameters:
    ///   - maxNoImprovementEvals: Stop after this many evaluations without improvement.
    ///   - minEvals: Minimum evaluations before stopping is considered.
    ///   - valueProvider: Closure that returns the current metric value.
    ///   - verbose: Verbosity level.
    public init(
        maxNoImprovementEvals: Int,
        minEvals: Int = 0,
        valueProvider: @escaping () -> Double?,
        verbose: Int = 1
    ) {
        self.maxNoImprovementEvals = maxNoImprovementEvals
        self.minEvals = minEvals
        self.valueProvider = valueProvider
        self.verbose = verbose
    }

    public func onStep(locals: CallbackLocals) -> Bool {
        guard let currentValue = valueProvider() else { return true }

        if lastValue == currentValue {
            return true
        }
        lastValue = currentValue
        evalCount += 1

        if currentValue > bestValue {
            bestValue = currentValue
            noImprovementCount = 0
        } else {
            noImprovementCount += 1
        }

        if evalCount >= minEvals && noImprovementCount >= maxNoImprovementEvals {
            stopped = true
            if verbose >= 1 {
                print(
                    "Stopping training: no improvement for \(noImprovementCount) evaluations. Best: \(String(format: "%.2f", bestValue))"
                )
            }
            return false
        }

        return true
    }
}

/// Callback that stops training after a specified number of timesteps.
public final class StopTrainingOnMaxTimesteps: Callback {
    public let maxTimesteps: Int
    public let verbose: Int

    private var stopped: Bool = false

    /// Whether training was stopped by this callback.
    public var hasStopped: Bool { stopped }

    /// Creates a stop training callback based on timestep limit.
    ///
    /// - Parameters:
    ///   - maxTimesteps: Stop after this many timesteps.
    ///   - verbose: Verbosity level.
    public init(maxTimesteps: Int, verbose: Int = 1) {
        self.maxTimesteps = maxTimesteps
        self.verbose = verbose
    }

    public func onStep(locals: CallbackLocals) -> Bool {
        if locals.numTimesteps >= maxTimesteps {
            stopped = true
            if verbose >= 1 {
                print("Stopping training: reached \(maxTimesteps) timesteps")
            }
            return false
        }
        return true
    }
}

/// Callback that stops training after a specified duration.
public final class StopTrainingOnTimeout: Callback {
    public let timeout: TimeInterval
    public let verbose: Int

    private var startTime: Date?
    private var stopped: Bool = false

    /// Whether training was stopped by this callback.
    public var hasStopped: Bool { stopped }

    /// Elapsed time since training started.
    public var elapsed: TimeInterval {
        guard let start = startTime else { return 0 }
        return Date().timeIntervalSince(start)
    }

    /// Creates a stop training callback based on time limit.
    ///
    /// - Parameters:
    ///   - timeout: Stop after this many seconds.
    ///   - verbose: Verbosity level.
    public init(timeout: TimeInterval, verbose: Int = 1) {
        self.timeout = timeout
        self.verbose = verbose
    }

    public func onTrainingStart(locals: CallbackLocals) {
        startTime = Date()
    }

    public func onStep(locals: CallbackLocals) -> Bool {
        guard let start = startTime else { return true }

        let elapsed = Date().timeIntervalSince(start)
        if elapsed >= timeout {
            stopped = true
            if verbose >= 1 {
                print("Stopping training: timeout after \(String(format: "%.1f", elapsed)) seconds")
            }
            return false
        }

        return true
    }
}
