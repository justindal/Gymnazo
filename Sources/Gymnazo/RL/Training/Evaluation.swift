//
//  Evaluation.swift
//  Gymnazo
//

import Foundation
import MLX

/// Result of evaluating a policy on an environment.
public struct EvaluationResult: Sendable {
    public let meanReward: Double
    public let stdReward: Double
    public let meanEpisodeLength: Double
    public let stdEpisodeLength: Double
    public let episodeRewards: [Double]
    public let episodeLengths: [Int]
    public let successRate: Double?

    public init(
        meanReward: Double,
        stdReward: Double,
        meanEpisodeLength: Double,
        stdEpisodeLength: Double,
        episodeRewards: [Double],
        episodeLengths: [Int],
        successRate: Double?
    ) {
        self.meanReward = meanReward
        self.stdReward = stdReward
        self.meanEpisodeLength = meanEpisodeLength
        self.stdEpisodeLength = stdEpisodeLength
        self.episodeRewards = episodeRewards
        self.episodeLengths = episodeLengths
        self.successRate = successRate
    }
}

/// Evaluates a policy that takes MLXArray observations and returns MLXArray actions.
///
/// - Parameters:
///   - policy: The policy to evaluate.
///   - env: The environment to evaluate on.
///   - nEvalEpisodes: Number of episodes to run.
///   - deterministic: Whether to use deterministic actions.
///   - render: Whether to render the environment during evaluation.
/// - Returns: Evaluation statistics.
public func evaluatePolicy<E: Env>(
    policy: some Policy,
    env: inout E,
    nEvalEpisodes: Int = 10,
    deterministic: Bool = true,
    render: Bool = false
) throws -> EvaluationResult where E.Observation == MLXArray, E.Action == MLXArray {
    var episodeRewards: [Double] = []
    var episodeLengths: [Int] = []
    var successes: [Bool] = []

    episodeRewards.reserveCapacity(nEvalEpisodes)
    episodeLengths.reserveCapacity(nEvalEpisodes)

    for _ in 0..<nEvalEpisodes {
        var obs = try env.reset().obs
        var done = false
        var episodeReward: Double = 0
        var episodeLength: Int = 0

        while !done {
            let action = policy.predict(observation: obs, deterministic: deterministic)
            MLX.eval(action)

            let step = try env.step(action)

            episodeReward += step.reward
            episodeLength += 1
            done = step.terminated || step.truncated
            obs = step.obs

            if let success = step.info["is_success"]?.bool {
                successes.append(success)
            }

            if render {
                try env.render()
            }
        }

        episodeRewards.append(episodeReward)
        episodeLengths.append(episodeLength)
    }

    let rewardStats = computeStats(episodeRewards)
    let lengthStats = computeStats(episodeLengths.map(Double.init))

    let successRate: Double? =
        successes.isEmpty ? nil : Double(successes.filter { $0 }.count) / Double(successes.count)

    return EvaluationResult(
        meanReward: rewardStats.mean,
        stdReward: rewardStats.std,
        meanEpisodeLength: lengthStats.mean,
        stdEpisodeLength: lengthStats.std,
        episodeRewards: episodeRewards,
        episodeLengths: episodeLengths,
        successRate: successRate
    )
}

/// Evaluates a policy that takes MLXArray observations and returns Int actions.
///
/// - Parameters:
///   - policy: The policy to evaluate.
///   - env: The environment to evaluate on.
///   - nEvalEpisodes: Number of episodes to run.
///   - deterministic: Whether to use deterministic actions.
///   - render: Whether to render the environment during evaluation.
/// - Returns: Evaluation statistics.
public func evaluatePolicy<E: Env>(
    policy: some Policy,
    env: inout E,
    nEvalEpisodes: Int = 10,
    deterministic: Bool = true,
    render: Bool = false
) throws -> EvaluationResult where E.Observation == MLXArray, E.Action == Int {
    var episodeRewards: [Double] = []
    var episodeLengths: [Int] = []
    var successes: [Bool] = []

    episodeRewards.reserveCapacity(nEvalEpisodes)
    episodeLengths.reserveCapacity(nEvalEpisodes)

    for _ in 0..<nEvalEpisodes {
        var obs = try env.reset().obs
        var done = false
        var episodeReward: Double = 0
        var episodeLength: Int = 0

        while !done {
            let actionArray = policy.predict(observation: obs, deterministic: deterministic)
            MLX.eval(actionArray)
            let action = Int(actionArray.item(Int32.self))

            let step = try env.step(action)

            episodeReward += step.reward
            episodeLength += 1
            done = step.terminated || step.truncated
            obs = step.obs

            if let success = step.info["is_success"]?.bool {
                successes.append(success)
            }

            if render {
                try env.render()
            }
        }

        episodeRewards.append(episodeReward)
        episodeLengths.append(episodeLength)
    }

    let rewardStats = computeStats(episodeRewards)
    let lengthStats = computeStats(episodeLengths.map(Double.init))

    let successRate: Double? =
        successes.isEmpty ? nil : Double(successes.filter { $0 }.count) / Double(successes.count)

    return EvaluationResult(
        meanReward: rewardStats.mean,
        stdReward: rewardStats.std,
        meanEpisodeLength: lengthStats.mean,
        stdEpisodeLength: lengthStats.std,
        episodeRewards: episodeRewards,
        episodeLengths: episodeLengths,
        successRate: successRate
    )
}

/// Tracks episode statistics during training.
public final class EpisodeMonitor {
    private var currentReward: Double = 0
    private var currentLength: Int = 0
    private var recentRewards: RingBuffer<Double>
    private var recentLengths: RingBuffer<Int>
    private var totalEpisodes: Int = 0

    /// Creates an episode monitor with a specified window size.
    ///
    /// - Parameter windowSize: Number of recent episodes to track for statistics.
    public init(windowSize: Int = 100) {
        self.recentRewards = RingBuffer(capacity: windowSize)
        self.recentLengths = RingBuffer(capacity: windowSize)
    }

    /// Records a step reward.
    ///
    /// - Parameter reward: The reward received at this step.
    public func step(reward: Double) {
        currentReward += reward
        currentLength += 1
    }

    /// Marks the end of an episode and returns its statistics.
    ///
    /// - Returns: Tuple of episode reward and length.
    @discardableResult
    public func episodeEnd() -> (reward: Double, length: Int) {
        let result = (currentReward, currentLength)
        recentRewards.append(currentReward)
        recentLengths.append(currentLength)
        totalEpisodes += 1
        currentReward = 0
        currentLength = 0
        return result
    }

    /// Resets the monitor state.
    public func reset() {
        currentReward = 0
        currentLength = 0
        recentRewards.clear()
        recentLengths.clear()
        totalEpisodes = 0
    }

    /// Mean reward over recent episodes.
    public var meanReward: Double {
        recentRewards.isEmpty ? 0 : recentRewards.reduce(0, +) / Double(recentRewards.count)
    }

    /// Standard deviation of reward over recent episodes.
    public var stdReward: Double {
        computeStats(Array(recentRewards)).std
    }

    /// Mean episode length over recent episodes.
    public var meanLength: Double {
        recentLengths.isEmpty ? 0 : Double(recentLengths.reduce(0, +)) / Double(recentLengths.count)
    }

    /// Number of episodes recorded.
    public var numEpisodes: Int { totalEpisodes }

    /// Number of episodes in the current window.
    public var windowCount: Int { recentRewards.count }
}

/// A ring buffer for storing recent values.
public struct RingBuffer<T>: Sequence {
    private var storage: [T?]
    private var head: Int = 0
    private var _count: Int = 0

    /// Creates a ring buffer with specified capacity.
    ///
    /// - Parameter capacity: Maximum number of elements.
    public init(capacity: Int) {
        precondition(capacity > 0)
        self.storage = Array(repeating: nil, count: capacity)
    }

    /// Appends an element, overwriting oldest if at capacity.
    ///
    /// - Parameter element: The element to append.
    public mutating func append(_ element: T) {
        storage[head] = element
        head = (head + 1) % storage.count
        if _count < storage.count {
            _count += 1
        }
    }

    /// Clears all elements.
    public mutating func clear() {
        storage = Array(repeating: nil, count: storage.count)
        head = 0
        _count = 0
    }

    /// Number of elements currently stored.
    public var count: Int { _count }

    /// Whether the buffer is empty.
    public var isEmpty: Bool { _count == 0 }

    /// Maximum capacity.
    public var capacity: Int { storage.count }

    public func makeIterator() -> AnyIterator<T> {
        var index = 0
        return AnyIterator {
            guard index < self._count else { return nil }
            let storageIndex =
                (self.head - self._count + index + self.storage.count) % self.storage.count
            index += 1
            return self.storage[storageIndex]
        }
    }
}

private func computeStats(_ values: [Double]) -> (mean: Double, std: Double) {
    guard !values.isEmpty else { return (0, 0) }

    let mean = values.reduce(0, +) / Double(values.count)

    guard values.count > 1 else { return (mean, 0) }

    let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(values.count)
    let std = variance.squareRoot()

    return (mean, std)
}
