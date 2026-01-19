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

        public init(
            bufferSize: Int,
            optimizeMemoryUsage: Bool = false,
            handleTimeoutTermination: Bool = true
        ) {
            precondition(bufferSize > 0, "bufferSize must be positive, got \(bufferSize)")
            self.bufferSize = bufferSize
            self.optimizeMemoryUsage = optimizeMemoryUsage
            self.handleTimeoutTermination = handleTimeoutTermination
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
    public let observationSpace: any Space<MLXArray>
    public let actionSpace: any Space<MLXArray>
    public let numEnvs: Int

    private var observations: [Obs?]
    private var nextObservations: [Obs?]?
    private var actions: [MLXArray?]
    private var rewards: [MLXArray?]
    private var dones: [MLXArray?]
    private var timeouts: [MLXArray?]

    private var position: Int = 0
    private var isBufferFull: Bool = false

    public var count: Int { isBufferFull ? bufferSize : position }

    public init(
        observationSpace: any Space<MLXArray>,
        actionSpace: any Space<MLXArray>,
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

    public func sample(_ batchSize: Int, key: MLXArray) -> Sample {
        precondition(batchSize > 0, "batchSize must be positive, got \(batchSize)")
        precondition(count >= batchSize, "Not enough samples")

        let indices = sampleIndices(batchSize: batchSize, key: key)

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

        var donesBatch = MLXArray.stack(batchDones)
        let timeoutsBatch = MLXArray.stack(batchTimeouts)
        if config.handleTimeoutTermination {
            donesBatch = donesBatch * (1.0 - timeoutsBatch)
        }

        return Sample(
            obs: Obs.stack(batchObs),
            actions: MLXArray.stack(batchActions),
            rewards: MLXArray.stack(batchRewards),
            nextObs: Obs.stack(batchNextObs),
            dones: donesBatch,
            timeouts: timeoutsBatch
        )
    }

    private func sampleIndices(batchSize: Int, key: MLXArray) -> [Int] {
        if !config.optimizeMemoryUsage {
            let indices = MLX.randInt(
                low: MLXArray(0),
                high: MLXArray(Int32(count)),
                [batchSize],
                key: key
            )
            MLX.eval(indices)
            return (0..<batchSize).map { i in indices[i].item(Int.self) }
        }

        if !isBufferFull {
            let indices = MLX.randInt(
                low: MLXArray(0),
                high: MLXArray(Int32(count)),
                [batchSize],
                key: key
            )
            MLX.eval(indices)
            return (0..<batchSize).map { i in indices[i].item(Int.self) }
        }

        let indices = MLX.randInt(
            low: MLXArray(0),
            high: MLXArray(Int32(bufferSize - 1)),
            [batchSize],
            key: key
        )
        MLX.eval(indices)
        return (0..<batchSize).map { i in
            var idx = indices[i].item(Int.self)
            if idx >= position { idx += 1 }
            return idx
        }
    }
}
