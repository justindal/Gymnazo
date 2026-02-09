import MLX

public struct ReplayBuffer: Buffer {
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
            precondition(bufferSize > 0)
            self.bufferSize = bufferSize
            self.optimizeMemoryUsage = optimizeMemoryUsage
            self.handleTimeoutTermination = handleTimeoutTermination
            self.seed = seed
        }
    }

    public struct Sample {
        public let obs: MLXArray
        public let actions: MLXArray
        public let rewards: MLXArray
        public let nextObs: MLXArray
        public let dones: MLXArray
        public let timeouts: MLXArray
    }

    public let config: Configuration
    public var bufferSize: Int { config.bufferSize }
    public let observationSpace: any Space
    public let actionSpace: any Space
    public let numEnvs: Int

    var observations: MLXArray
    var nextObservations: MLXArray?
    var actions: MLXArray
    var rewards: MLXArray
    var dones: MLXArray
    var timeouts: MLXArray

    var position: Int = 0
    var isBufferFull: Bool = false
    private var seededRng: SplitMix64?
    private var systemRng = SystemRandomNumberGenerator()

    public var count: Int { isBufferFull ? bufferSize : position }

    public init(
        observationSpace: any Space,
        actionSpace: any Space,
        config: Configuration,
        numEnvs: Int = 1
    ) {
        precondition(numEnvs > 0)
        precondition(
            !(config.optimizeMemoryUsage && config.handleTimeoutTermination),
            "optimizeMemoryUsage is not compatible with handleTimeoutTermination")
        if config.optimizeMemoryUsage {
            precondition(config.bufferSize > 1)
        }

        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        self.config = config
        self.numEnvs = numEnvs

        let (obs, nextObs, acts, rews, dns, tos) = Self.allocateStorage(
            bufferSize: config.bufferSize,
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            optimizeMemory: config.optimizeMemoryUsage
        )
        self.observations = obs
        self.nextObservations = nextObs
        self.actions = acts
        self.rewards = rews
        self.dones = dns
        self.timeouts = tos

        var toEval = [observations, actions, rewards, dones, timeouts]
        if let next = nextObservations { toEval.append(next) }
        eval(toEval)

        if let seed = config.seed {
            self.seededRng = SplitMix64(state: seed)
        }
    }

    public mutating func reset() {
        let (obs, nextObs, acts, rews, dns, tos) = Self.allocateStorage(
            bufferSize: config.bufferSize,
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            optimizeMemory: config.optimizeMemoryUsage
        )
        observations = obs
        nextObservations = nextObs
        actions = acts
        rewards = rews
        dones = dns
        timeouts = tos

        var toEval = [observations, actions, rewards, dones, timeouts]
        if let next = nextObservations { toEval.append(next) }
        eval(toEval)

        position = 0
        isBufferFull = false
    }

    public mutating func add(
        obs: MLXArray,
        action: MLXArray,
        reward: MLXArray,
        nextObs: MLXArray,
        terminated: Bool,
        truncated: Bool
    ) {
        let doneValue: Float = (terminated || truncated) ? 1.0 : 0.0
        let timeoutValue: Float = truncated ? 1.0 : 0.0

        observations[position] = obs
        actions[position] = action.reshaped([-1])
        rewards[position] = reward
        dones[position] = MLXArray(doneValue)
        timeouts[position] = MLXArray(timeoutValue)

        if config.optimizeMemoryUsage {
            let nextPos = (position + 1) % bufferSize
            observations[nextPos] = nextObs
        } else {
            nextObservations![position] = nextObs
        }

        var toEval = [observations, actions, rewards, dones, timeouts]
        if let next = nextObservations { toEval.append(next) }
        eval(toEval)

        position = (position + 1) % bufferSize
        if position == 0 { isBufferFull = true }
    }

    public mutating func sample(_ batchSize: Int, key: MLXArray) -> Sample {
        precondition(batchSize > 0)
        precondition(count >= batchSize)

        let rawIndices = sampleIndices(batchSize: batchSize)
        let indices = MLXArray(rawIndices.map { Int32($0) })

        let obsBatch = observations[indices]
        let actionsBatch = actions[indices]
        let rewardsBatch = rewards[indices]
        var donesBatch = dones[indices]
        let timeoutsBatch = timeouts[indices]

        let nextObsBatch: MLXArray
        if config.optimizeMemoryUsage {
            let nextIndices = MLXArray(rawIndices.map { Int32(($0 + 1) % bufferSize) })
            nextObsBatch = observations[nextIndices]
        } else {
            nextObsBatch = nextObservations![indices]
        }

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

    private static func allocateStorage(
        bufferSize: Int,
        observationSpace: any Space,
        actionSpace: any Space,
        optimizeMemory: Bool
    ) -> (MLXArray, MLXArray?, MLXArray, MLXArray, MLXArray, MLXArray) {
        let obsShape = observationSpace.shape ?? [1]
        let obsDtype = observationSpace.dtype ?? .float32
        let actionDim = bufferActionDim(for: actionSpace)

        let obs = MLX.zeros([bufferSize] + obsShape, dtype: obsDtype)
        let nextObs = optimizeMemory
            ? nil : MLX.zeros([bufferSize] + obsShape, dtype: obsDtype)
        let acts = MLX.zeros([bufferSize, actionDim])
        let rews = MLX.zeros([bufferSize])
        let dns = MLX.zeros([bufferSize])
        let tos = MLX.zeros([bufferSize])

        return (obs, nextObs, acts, rews, dns, tos)
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

private func bufferActionDim(for actionSpace: any Space) -> Int {
    (actionSpace as? Box)?.shape?.reduce(1, *) ?? 1
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
