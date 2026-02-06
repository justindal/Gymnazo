//
//  Algorithm.swift
//  Gymnazo
//

/// Learning rate schedule evaluated with progress remaining in [0, 1].
public protocol LearningRateSchedule {
    func value(at progressRemaining: Double) -> Double
}

/// Constant learning rate schedule.
public struct ConstantLearningRate: LearningRateSchedule {
    public let value: Double

    public init(_ value: Double) {
        self.value = value
    }

    public func value(at progressRemaining: Double) -> Double {
        value
    }
}

/// Callbacks for the learn() method to enable UI updates and control flow.
public struct LearnCallbacks: Sendable {
    public typealias OnStepCallback = @Sendable (Int, Int, Double) -> Bool
    public typealias OnEpisodeEndCallback = @Sendable (Double, Int) -> Void
    public typealias OnSnapshotCallback = @Sendable (any Sendable) -> Void

    public var onStep: OnStepCallback?
    public var onEpisodeEnd: OnEpisodeEndCallback?
    public var onSnapshot: OnSnapshotCallback?

    public init(
        onStep: OnStepCallback? = nil,
        onEpisodeEnd: OnEpisodeEndCallback? = nil,
        onSnapshot: OnSnapshotCallback? = nil
    ) {
        self.onStep = onStep
        self.onEpisodeEnd = onEpisodeEnd
        self.onSnapshot = onSnapshot
    }
}

/// Base protocol for reinforcement learning algorithms.
public protocol Algorithm: AnyObject {
    associatedtype PolicyType: Policy

    var policy: PolicyType { get }
    var env: (any Env)? { get set }

    var learningRate: any LearningRateSchedule { get }
    var currentProgressRemaining: Double { get set }

    var numTimesteps: Int { get set }
    var totalTimesteps: Int { get set }

    @discardableResult
    func learn(totalTimesteps: Int) throws -> Self

    @discardableResult
    func learn(totalTimesteps: Int, callbacks: LearnCallbacks?) throws -> Self
}

extension Algorithm {
    public var unwrappedEnv: (any Env)? { env?.unwrapped }

    public func updateProgressRemaining(numTimesteps: Int, totalTimesteps: Int) {
        currentProgressRemaining = 1.0 - Double(numTimesteps) / Double(totalTimesteps)
    }

    public func currentLearningRate() -> Double {
        learningRate.value(at: currentProgressRemaining)
    }
}
