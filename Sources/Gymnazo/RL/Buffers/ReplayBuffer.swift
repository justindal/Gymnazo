import MLX

public protocol Batchable {
    static func stack(_ items: [Self]) -> Self
}

extension MLXArray: Batchable {
    public static func stack(_ items: [MLXArray]) -> MLXArray {
        MLX.stacked(items)
    }
}

extension Dictionary: Batchable where Key == String, Value == MLXArray {
    public static func stack(_ items: [Self]) -> Self {
        guard let first: [String: MLXArray] = items.first else { return [:] }
        var result: [String: MLXArray] = [:]
        for key: String in first.keys {
            result[key] = MLX.stacked(items.compactMap { $0[key] })
        }
        return result
    }
}

/// A replay buffer for off-policy RL algorithms.
public struct ReplayBuffer<Obs: Batchable>: Buffer {
    public struct Configuration: Sendable {
        public let bufferSize: Int
        public let optimizeMemoryUsage: Bool
        public let handleTimeoutTermination: Bool
        public let seed: UInt64?

        public init(
            bufferSize: Int,
            optimizeMemoryUsage: Bool = false,
            handleTimeoutTermination: Bool = true,
            seed: UInt64? = nil
        ) {
            precondition(bufferSize > 0, "bufferSize must be positive, got \(bufferSize)")
            self.bufferSize = bufferSize
            self.optimizeMemoryUsage = optimizeMemoryUsage
            self.handleTimeoutTermination = handleTimeoutTermination
            self.seed = seed
        }
    }

    public struct Sample {
        public let obs: Obs
        public let actions: MLXArray
        public let rewards: MLXArray
        public let nextObs: Obs
        public let dones: MLXArray
        public let timeouts: MLXArray
    }

    public let config: Configuration

    public var bufferSize: Int { config.bufferSize }
    public let observationSpace: any Space
    public let actionSpace: any Space
    public let numEnvs: Int

    private var observations: [Obs?]
    private var nextObservations: [Obs?]?
    private var actions: [MLXArray?]
    private var rewards: [MLXArray?]
    private var dones: [MLXArray?]
    private var timeouts: [MLXArray?]

    private var position: Int = 0
    private var isBufferFull: Bool = false
    private var seededRng: SplitMix64?
    private var systemRng = SystemRandomNumberGenerator()

    public var count: Int { isBufferFull ? bufferSize : position }

    public init(
        observationSpace: any Space,
        actionSpace: any Space,
        config: Configuration,
        numEnvs: Int = 1
    ) {
        precondition(numEnvs > 0, "numEnvs must be positive, got \(numEnvs)")
        precondition(
            !(config.optimizeMemoryUsage && config.handleTimeoutTermination),
            "optimizeMemoryUsage is not compatible with handleTimeoutTermination")
        if config.optimizeMemoryUsage {
            precondition(
                config.bufferSize > 1,
                "optimizeMemoryUsage requires bufferSize > 1, got \(config.bufferSize)")
            precondition(
                !(observationSpace is Dict),
                "optimizeMemoryUsage is not compatible with Dict observations")
        }

        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        self.config = config
        self.numEnvs = numEnvs

        let n = config.bufferSize
        self.observations = Array(repeating: nil, count: n)
        self.nextObservations = config.optimizeMemoryUsage ? nil : Array(repeating: nil, count: n)
        self.actions = Array(repeating: nil, count: n)
        self.rewards = Array(repeating: nil, count: n)
        self.dones = Array(repeating: nil, count: n)
        self.timeouts = Array(repeating: nil, count: n)
        if let seed = config.seed {
            self.seededRng = SplitMix64(state: seed)
        }
    }

    public mutating func reset() {
        let n = config.bufferSize
        observations = Array(repeating: nil, count: n)
        nextObservations = config.optimizeMemoryUsage ? nil : Array(repeating: nil, count: n)
        actions = Array(repeating: nil, count: n)
        rewards = Array(repeating: nil, count: n)
        dones = Array(repeating: nil, count: n)
        timeouts = Array(repeating: nil, count: n)
        position = 0
        isBufferFull = false
    }

    public mutating func add(
        obs: Obs,
        action: MLXArray,
        reward: MLXArray,
        nextObs: Obs,
        terminated: Bool,
        truncated: Bool
    ) {
        let doneValue: Float = (terminated || truncated) ? 1.0 : 0.0
        let timeoutValue: Float = truncated ? 1.0 : 0.0

        observations[position] = obs
        actions[position] = action
        rewards[position] = reward
        dones[position] = MLXArray(doneValue)
        timeouts[position] = MLXArray(timeoutValue)

        let nextPosition = (position + 1) % bufferSize
        if config.optimizeMemoryUsage {
            observations[nextPosition] = nextObs
        } else {
            nextObservations![position] = nextObs
        }

        position = nextPosition
        if position == 0 { isBufferFull = true }
    }

    public mutating func sample(_ batchSize: Int, key: MLXArray) -> Sample {
        precondition(batchSize > 0, "batchSize must be positive, got \(batchSize)")
        precondition(count >= batchSize, "Not enough samples")

        let indices = sampleIndices(batchSize: batchSize)

        var batchObs: [Obs] = []
        var batchActions: [MLXArray] = []
        var batchRewards: [MLXArray] = []
        var batchNextObs: [Obs] = []
        var batchDones: [MLXArray] = []
        var batchTimeouts: [MLXArray] = []

        batchObs.reserveCapacity(batchSize)
        batchActions.reserveCapacity(batchSize)
        batchRewards.reserveCapacity(batchSize)
        batchNextObs.reserveCapacity(batchSize)
        batchDones.reserveCapacity(batchSize)
        batchTimeouts.reserveCapacity(batchSize)

        for idx in indices {
            guard
                let obs = observations[idx],
                let action = actions[idx],
                let reward = rewards[idx],
                let done = dones[idx],
                let timeout = timeouts[idx]
            else {
                preconditionFailure(
                    "ReplayBuffer sampled an uninitialized transition at index \(idx)")
            }

            batchObs.append(obs)
            batchActions.append(action)
            batchRewards.append(reward)
            batchDones.append(done)
            batchTimeouts.append(timeout)

            if config.optimizeMemoryUsage {
                let nextIdx = (idx + 1) % bufferSize
                guard let nextObs = observations[nextIdx] else {
                    preconditionFailure(
                        "ReplayBuffer sampled an uninitialized next observation at index \(nextIdx)"
                    )
                }
                batchNextObs.append(nextObs)
            } else {
                guard let nextObs = nextObservations?[idx] else {
                    preconditionFailure(
                        "ReplayBuffer sampled an uninitialized next observation at index \(idx)")
                }
                batchNextObs.append(nextObs)
            }
        }

        let obsBatch = Obs.stack(batchObs)
        let actionsBatch = MLXArray.stack(batchActions)
        let rewardsBatch = MLXArray.stack(batchRewards)
        let nextObsBatch = Obs.stack(batchNextObs)
        var donesBatch = MLXArray.stack(batchDones)
        let timeoutsBatch = MLXArray.stack(batchTimeouts)
        if config.handleTimeoutTermination {
            donesBatch = donesBatch * (1.0 - timeoutsBatch)
        }

        return Sample(
            obs: obsBatch,
            actions: actionsBatch,
            rewards: rewardsBatch,
            nextObs: nextObsBatch,
            dones: donesBatch,
            timeouts: timeoutsBatch
        )
    }

    private mutating func sampleIndices(batchSize: Int) -> [Int] {
        var indices: [Int] = []
        indices.reserveCapacity(batchSize)

        if !config.optimizeMemoryUsage || !isBufferFull {
            let upper = count
            for _ in 0..<batchSize {
                indices.append(nextIndex(upperBound: upper))
            }
            return indices
        }

        let upper = bufferSize - 1
        for _ in 0..<batchSize {
            var idx = nextIndex(upperBound: upper)
            if idx >= position { idx += 1 }
            indices.append(idx)
        }
        return indices
    }

    private mutating func nextIndex(upperBound: Int) -> Int {
        if var rng = seededRng {
            let value = rng.next()
            seededRng = rng
            return Int(value % UInt64(upperBound))
        }
        return Int(systemRng.next() % UInt64(upperBound))
    }
}

private struct SplitMix64: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
