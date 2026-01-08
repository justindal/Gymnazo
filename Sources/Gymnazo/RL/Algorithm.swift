//
//  Algorithm.swift
//  Gymnazo
//

/// Learning rate schedule evaluated with progress remaining in [0, 1].
public protocol LearningRateSchedule {
    func value(at progressRemaining: Double) -> Double
    // https://www.hackingwithswift.com/swift/5.2/callasfunction im not sure if this is appropriate here but TODO
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

/// Base algorithm protocol
public protocol Algorithm {
    associatedtype PolicyType: Policy
    associatedtype EnvType: Env

    var policy: PolicyType { get }
    var env: EnvType? { get set }

    var learningRate: any LearningRateSchedule { get }
    var currentProgressRemaining: Double { get set }

    var numTimesteps: Int { get set }
    var totalTimesteps: Int { get set }

    mutating func setupModel()

    @discardableResult
    mutating func learn(totalTimesteps: Int) -> Self
}

extension Algorithm {
    public mutating func updateProgressRemaining(numTimesteps: Int, totalTimesteps: Int) {
        currentProgressRemaining = 1.0 - Double(numTimesteps) / Double(totalTimesteps)
    }

    public func currentLearningRate() -> Double {
        learningRate.value(at: currentProgressRemaining)
    }
}
