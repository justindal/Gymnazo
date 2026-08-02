import Foundation

public struct RecordEpisodeStatisticsValueEnvironment<Base: ValueEnvironment>: ValueEnvironment {
    public typealias Observation = Base.Observation
    public typealias Action = Base.Action
    public typealias ObservationSpace = Base.ObservationSpace
    public typealias ActionSpace = Base.ActionSpace

    public var base: Base
    public private(set) var episodeCount = 0
    public private(set) var timeQueue: [Double] = []
    public private(set) var returnQueue: [Double] = []
    public private(set) var lengthQueue: [Int] = []

    private let bufferLength: Int
    private let statsKey: String
    private var episodeStartTime = ProcessInfo.processInfo.systemUptime
    private var episodeReturn = 0.0
    private var episodeLength = 0

    public var valueObservationSpace: ObservationSpace {
        base.valueObservationSpace
    }

    public var valueActionSpace: ActionSpace {
        base.valueActionSpace
    }

    public init(
        _ base: Base,
        bufferLength: Int = 100,
        statsKey: String = EnvInfoKey.episode
    ) throws {
        guard bufferLength > 0 else {
            throw GymnazoError.invalidRecordBufferLength(bufferLength)
        }
        self.base = base
        self.bufferLength = bufferLength
        self.statsKey = statsKey
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation> {
        let result = try base.resetValue(seed: seed, options: options)
        episodeStartTime = ProcessInfo.processInfo.systemUptime
        episodeReturn = 0
        episodeLength = 0
        return result
    }

    public mutating func stepValue(_ action: Action) throws -> ValueStep<Observation> {
        let result = try base.stepValue(action)
        episodeReturn += result.reward
        episodeLength += 1
        guard result.terminated || result.truncated else {
            return result
        }
        guard result.info[statsKey] == nil else {
            throw GymnazoError.invalidStatsKey(statsKey)
        }

        let elapsed = Self.rounded(ProcessInfo.processInfo.systemUptime - episodeStartTime)
        var info = result.info
        info[statsKey] = .dictionary([
            "r": .double(episodeReturn),
            "l": .discrete(episodeLength),
            "t": .double(elapsed),
        ])
        append(elapsed, to: &timeQueue)
        append(episodeReturn, to: &returnQueue)
        append(episodeLength, to: &lengthQueue)
        episodeCount += 1
        return ValueStep(
            observation: result.observation,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: info
        )
    }

    public mutating func close() {
        base.close()
    }

    private func append<T>(_ value: T, to queue: inout [T]) {
        queue.append(value)
        if queue.count > bufferLength {
            queue.removeFirst(queue.count - bufferLength)
        }
    }

    private static func rounded(_ value: Double) -> Double {
        (value * 1_000_000).rounded() / 1_000_000
    }
}
