/// Learning rate schedule evaluated with progress remaining in [0, 1].
public protocol LearningRateSchedule: Sendable {
    func value(at progressRemaining: Double) -> Double
}

/// Constant learning rate schedule.
public struct ConstantLearningRate: LearningRateSchedule, Codable, Sendable {
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
    public typealias OnStepCallback = @Sendable (Int, Int, Double) async -> Bool
    public typealias OnEpisodeEndCallback = @Sendable (Double, Int) async -> Void
    public typealias OnSnapshotCallback = @Sendable (any Sendable) async -> Void
    public typealias OnTrainCallback = @Sendable ([String: Double]) async -> Void

    public var onStep: OnStepCallback?
    public var onEpisodeEnd: OnEpisodeEndCallback?
    public var onSnapshot: OnSnapshotCallback?
    public var onTrain: OnTrainCallback?

    public init(
        onStep: OnStepCallback? = nil,
        onEpisodeEnd: OnEpisodeEndCallback? = nil,
        onSnapshot: OnSnapshotCallback? = nil,
        onTrain: OnTrainCallback? = nil
    ) {
        self.onStep = onStep
        self.onEpisodeEnd = onEpisodeEnd
        self.onSnapshot = onSnapshot
        self.onTrain = onTrain
    }
}

/// Callbacks for the evaluate() method to enable UI updates and control flow.
public struct EvaluateCallbacks: Sendable {
    public typealias OnStepCallback = @Sendable () async -> Bool
    public typealias OnEpisodeEndCallback = @Sendable (Double, Int) async -> Void
    public typealias OnSnapshotCallback = @Sendable (any Sendable) async -> Void

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

