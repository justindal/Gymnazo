//
// RecordEpisodeStatistics.swift
//

import Foundation

/// Tracks cumulative rewards, episode lengths, and elapsed time, attaching an "episode"
/// entry to the `info` dictionary when an episode terminates or truncates.
public final class RecordEpisodeStatistics<InnerEnv: Env>: Wrapper {
    public typealias Observation = InnerEnv.Observation
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = InnerEnv.ObservationSpace
    public typealias ActionSpace = InnerEnv.ActionSpace

    public var env: InnerEnv

    public private(set) var episodeCount: Int = 0
    public private(set) var timeQueue: [Double] = []
    public private(set) var returnQueue: [Double] = []
    public private(set) var lengthQueue: [Int] = []

    private let bufferLength: Int
    private let statsKey: String

    private var episodeStartTime: TimeInterval = RecordEpisodeStatistics.now()
    private var episodeReturns: Double = 0.0
    private var episodeLengths: Int = 0

    public required convenience init(env: InnerEnv) {
        self.init(env: env, bufferLength: 100, statsKey: "episode")
    }

    public init(env: InnerEnv, bufferLength: Int = 100, statsKey: String = "episode") {
        precondition(bufferLength > 0, "bufferLength must be positive, got \(bufferLength)")
        self.env = env
        self.bufferLength = bufferLength
        self.statsKey = statsKey
    }

    public func reset(seed: UInt64?, options: [String : Any]?) -> Reset<Observation> {
        let result = env.reset(seed: seed, options: options)
        episodeStartTime = RecordEpisodeStatistics.now()
        episodeReturns = 0.0
        episodeLengths = 0
        return result
    }

    public func step(_ action: Action) -> Step<Observation> {
        let result = env.step(action)

        episodeReturns += result.reward
        episodeLengths += 1

        var info = result.info

        if result.terminated || result.truncated {
            precondition(info[statsKey] == nil, "RecordEpisodeStatistics: info already contains key \(statsKey)")

            let now = RecordEpisodeStatistics.now()
            let elapsed = RecordEpisodeStatistics.roundTime(now - episodeStartTime)

            info[statsKey] = .object([
                "r": .double(episodeReturns),
                "l": .int(episodeLengths),
                "t": .double(elapsed),
            ])

            append(elapsed, to: &timeQueue)
            append(episodeReturns, to: &returnQueue)
            append(episodeLengths, to: &lengthQueue)

            episodeCount += 1
            episodeStartTime = now
        }

        return Step(
            obs: result.obs,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: info
        )
    }

    private func append<T>(_ value: T, to queue: inout [T]) {
        queue.append(value)
        if queue.count > bufferLength {
            queue.removeFirst(queue.count - bufferLength)
        }
    }

    private static func now() -> TimeInterval {
        ProcessInfo.processInfo.systemUptime
    }

    private static func roundTime(_ value: Double) -> Double {
        let precision = 1_000_000.0
        return (value * precision).rounded() / precision
    }
}
