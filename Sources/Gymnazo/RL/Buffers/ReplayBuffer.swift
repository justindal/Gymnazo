import MLX

public struct ReplayBuffer: Buffer {
    public struct Configuration: Sendable {
        public enum FrameStackPadding: String, Sendable, Codable {
            case reset
            case zero
        }

        public struct FrameStackConfig: Sendable, Codable, Equatable {
            public let size: Int
            public let padding: FrameStackPadding
            public let axis: Int

            public init(size: Int, padding: FrameStackPadding = .reset, axis: Int = 0) {
                precondition(size > 1)
                self.size = size
                self.padding = padding
                self.axis = axis
            }

            public init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                size = try container.decode(Int.self, forKey: .size)
                padding = try container.decode(FrameStackPadding.self, forKey: .padding)
                axis = try container.decodeIfPresent(Int.self, forKey: .axis) ?? 0
            }

            public func resolvedAxis(ndim: Int) -> Int {
                axis >= 0 ? axis : ndim + axis
            }
        }

        public let bufferSize: Int
        public let optimizeMemoryUsage: Bool
        public let handleTimeoutTermination: Bool
        public let frameStack: FrameStackConfig?
        public let seed: UInt64?
        public let memoryBudgetBytes: Int?

        public init(
            bufferSize: Int,
            optimizeMemoryUsage: Bool = false,
            handleTimeoutTermination: Bool = true,
            frameStack: FrameStackConfig? = nil,
            seed: UInt64? = nil,
            memoryBudgetBytes: Int? = nil
        ) {
            precondition(bufferSize > 0)
            precondition(memoryBudgetBytes == nil || memoryBudgetBytes! > 0)
            self.bufferSize = bufferSize
            self.optimizeMemoryUsage = optimizeMemoryUsage
            self.handleTimeoutTermination = handleTimeoutTermination
            self.frameStack = frameStack
            self.seed = seed
            self.memoryBudgetBytes = memoryBudgetBytes
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
    public let allocationInfo: ReplayBufferAllocationInfo

    var observations: MLXArray
    var nextObservations: MLXArray?
    var actions: MLXArray
    var rewards: MLXArray
    var dones: MLXArray
    var timeouts: MLXArray
    var doneValues: [Bool]

    var position: Int = 0
    var isBufferFull: Bool = false
    private var randomGenerator: SeededGenerator
    let storageObsShape: [Int]
    private let storageObsDtype: DType
    private let actionDim: Int
    private let resolvedStackAxis: Int

    public var count: Int { isBufferFull ? bufferSize : position }

    public init(
        observationSpace: any Space,
        actionSpace: any Space,
        config: Configuration,
        numEnvs: Int = 1
    ) throws {
        precondition(numEnvs > 0)
        precondition(
            !(config.optimizeMemoryUsage && config.handleTimeoutTermination),
            "optimizeMemoryUsage is not compatible with handleTimeoutTermination")
        if config.optimizeMemoryUsage {
            precondition(config.bufferSize > 1)
        }

        let storageObsShape = Self.storageObservationShape(
            observationSpace: observationSpace,
            frameStack: config.frameStack
        )
        let storageObsDtype = observationSpace.dtype ?? .float32
        let actionDim = bufferActionDim(for: actionSpace)
        let resolved = try ReplayBufferSizing.resolve(
            config: config,
            storageObsShape: storageObsShape,
            storageObsDtype: storageObsDtype,
            actionDim: actionDim
        )

        let resolvedAxis: Int
        if let frameStack = resolved.config.frameStack,
            let obsNdim = observationSpace.shape?.count
        {
            resolvedAxis = frameStack.resolvedAxis(ndim: obsNdim)
        } else {
            resolvedAxis = 0
        }

        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        self.config = resolved.config
        self.allocationInfo = resolved.info
        self.numEnvs = numEnvs
        self.storageObsShape = storageObsShape
        self.storageObsDtype = storageObsDtype
        self.actionDim = actionDim
        self.resolvedStackAxis = resolvedAxis

        let (obs, nextObs, acts, rews, dns, tos) = Self.allocateStorage(
            bufferSize: resolved.config.bufferSize,
            observationShape: storageObsShape,
            observationDtype: storageObsDtype,
            actionDim: actionDim,
            optimizeMemory: resolved.config.optimizeMemoryUsage
        )
        self.observations = obs
        self.nextObservations = nextObs
        self.actions = acts
        self.rewards = rews
        self.dones = dns
        self.timeouts = tos
        self.doneValues = Array(repeating: false, count: resolved.config.bufferSize)
        self.randomGenerator = SeededGenerator(
            seed: resolved.config.seed ?? UInt64.random(in: 0...UInt64.max))
    }

    public mutating func reset() {
        let (obs, nextObs, acts, rews, dns, tos) = Self.allocateStorage(
            bufferSize: config.bufferSize,
            observationShape: storageObsShape,
            observationDtype: storageObsDtype,
            actionDim: actionDim,
            optimizeMemory: config.optimizeMemoryUsage
        )
        observations = obs
        nextObservations = nextObs
        actions = acts
        rewards = rews
        dones = dns
        timeouts = tos
        doneValues = Array(repeating: false, count: config.bufferSize)

        position = 0
        isBufferFull = false
        randomGenerator = SeededGenerator(seed: config.seed ?? UInt64.random(in: 0...UInt64.max))
    }

    public mutating func add(
        obs: MLXArray,
        action: MLXArray,
        reward: MLXArray,
        nextObs: MLXArray,
        terminated: Bool,
        truncated: Bool
    ) throws {
        let storedObs = toStoredObservation(obs)
        let storedNextObs = toStoredObservation(nextObs)
        try validateStoredObs(storedObs)
        try validateStoredObs(storedNextObs)
        try validateAction(action)

        let doneValue: Float = (terminated || truncated) ? 1.0 : 0.0
        let timeoutValue: Float = truncated ? 1.0 : 0.0

        observations[position] = storedObs
        actions[position] = action.reshaped([-1])
        rewards[position] = reward
        dones[position] = MLXArray(doneValue)
        timeouts[position] = MLXArray(timeoutValue)
        doneValues[position] = terminated || truncated

        if config.optimizeMemoryUsage {
            let nextPos = (position + 1) % bufferSize
            observations[nextPos] = storedNextObs
        } else {
            nextObservations![position] = storedNextObs
        }

        position = (position + 1) % bufferSize
        if position == 0 { isBufferFull = true }
    }

    public mutating func sample(_ batchSize: Int) throws -> Sample {
        precondition(batchSize > 0)
        precondition(count >= batchSize)
        let rawIndices = try sampleIndices(
            batchSize: batchSize,
            generator: &randomGenerator
        )
        return try sample(indices: rawIndices)
    }

    public mutating func sample(_ batchSize: Int, key: MLXArray) throws -> Sample {
        precondition(batchSize > 0)
        precondition(count >= batchSize)

        let keyValues = key.asArray(UInt32.self)
        let seed = keyValues.reduce(UInt64(0)) { partialResult, value in
            (partialResult << 32) ^ UInt64(value)
        }
        var generator = SeededGenerator(seed: seed)
        let rawIndices = try sampleIndices(
            batchSize: batchSize,
            generator: &generator
        )
        return try sample(indices: rawIndices)
    }

    private func sample(indices rawIndices: [Int]) throws -> Sample {
        let indices = MLXArray(rawIndices.map { Int32($0) })

        let actionsBatch = actions[indices]
        let rewardsBatch = rewards[indices]
        var donesBatch = dones[indices]
        let timeoutsBatch = timeouts[indices]

        let obsBatch: MLXArray
        let nextObsBatch: MLXArray
        if config.frameStack == nil {
            obsBatch = observations[indices]
            if config.optimizeMemoryUsage {
                let nextIndices = MLXArray(rawIndices.map { Int32(($0 + 1) % bufferSize) })
                nextObsBatch = observations[nextIndices]
            } else {
                nextObsBatch = nextObservations![indices]
            }
        } else {
            (obsBatch, nextObsBatch) = try sampleFrameStacks(rawIndices: rawIndices)
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

    private func sampleFrameStacks(rawIndices: [Int]) throws -> (MLXArray, MLXArray) {
        let frameStack = config.frameStack!
        let stackSize = frameStack.size
        let batchCount = rawIndices.count
        let frameShape = storageObsShape
        let isZeroPad = frameStack.padding == .zero

        var allFrameIndices: [Int32] = []
        allFrameIndices.reserveCapacity(batchCount * stackSize)
        var padMask: [Float]? = isZeroPad ? [] : nil
        padMask?.reserveCapacity(batchCount * stackSize)
        var nextFrameIndices: [Int32] = []
        nextFrameIndices.reserveCapacity(batchCount)

        for index in rawIndices {
            var frameIdxForSample: [Int] = []
            frameIdxForSample.reserveCapacity(stackSize)
            var padCount = 0

            var cursor = index
            frameIdxForSample.append(cursor)

            while frameIdxForSample.count < stackSize {
                if isEpisodeStartCPU(cursor) {
                    padCount = stackSize - frameIdxForSample.count
                    let padIdx = frameIdxForSample[frameIdxForSample.count - 1]
                    while frameIdxForSample.count < stackSize {
                        frameIdxForSample.append(padIdx)
                    }
                    break
                }
                cursor = previousIndex(cursor)
                frameIdxForSample.append(cursor)
            }

            frameIdxForSample.reverse()
            allFrameIndices.append(contentsOf: frameIdxForSample.map { Int32($0) })

            if isZeroPad {
                for i in 0..<stackSize {
                    padMask!.append(i < padCount ? 0.0 : 1.0)
                }
            }

            if config.optimizeMemoryUsage {
                nextFrameIndices.append(Int32((index + 1) % bufferSize))
            } else {
                nextFrameIndices.append(Int32(index))
            }
        }

        let gatherIndices = MLXArray(allFrameIndices)
        var allFrames = observations[gatherIndices]

        if let mask = padMask {
            let maskShape = [batchCount * stackSize] + Array(repeating: 1, count: frameShape.count)
            let maskArr = MLXArray(mask).reshaped(maskShape)
            allFrames = allFrames * maskArr
        }

        let batchedFrames = allFrames.reshaped([batchCount, stackSize] + frameShape)
        let obsBatch: MLXArray
        if resolvedStackAxis == 0 {
            obsBatch = batchedFrames
        } else {
            var perm = [0]
            let frameDims = Array(2..<(2 + frameShape.count))
            for (offset, dim) in frameDims.enumerated() {
                if offset == resolvedStackAxis {
                    perm.append(1)
                }
                perm.append(dim)
            }
            if perm.count < batchedFrames.ndim {
                perm.append(1)
            }
            obsBatch = batchedFrames.transposed(axes: perm)
        }

        let nextIndicesArr = MLXArray(nextFrameIndices)
        let nextFrames: MLXArray
        if config.optimizeMemoryUsage {
            nextFrames = observations[nextIndicesArr]
        } else {
            nextFrames = nextObservations![nextIndicesArr]
        }

        let nextObsBatch: MLXArray
        if resolvedStackAxis == 0 {
            let tail = obsBatch[0..., 1..<stackSize]
            let newFrame = nextFrames.expandedDimensions(axis: 1)
            nextObsBatch = MLX.concatenated([tail, newFrame], axis: 1)
        } else {
            let moved = obsBatch.swappedAxes(1, resolvedStackAxis + 1)
            let tail = moved[0..., 1..<stackSize]
            let newFrame = nextFrames.expandedDimensions(axis: 1)
            let shifted = MLX.concatenated([tail, newFrame], axis: 1)
            nextObsBatch = shifted.swappedAxes(1, resolvedStackAxis + 1)
        }

        return (obsBatch, nextObsBatch)
    }

    private func isEpisodeStartCPU(_ index: Int) -> Bool {
        if !isBufferFull {
            if index == 0 { return true }
            return doneValues[index - 1]
        }
        if index == position { return true }
        let previous = previousIndex(index)
        return doneValues[previous]
    }

    private static func storageObservationShape(
        observationSpace: any Space,
        frameStack: Configuration.FrameStackConfig?
    ) -> [Int] {
        let shape = observationSpace.shape ?? [1]
        guard let frameStack else { return shape }
        precondition(!shape.isEmpty)
        let axis = frameStack.resolvedAxis(ndim: shape.count)
        precondition(
            shape[axis] == frameStack.size,
            "frameStack size \(frameStack.size) does not match observation shape \(shape) at axis \(axis)"
        )
        var result = shape
        result.remove(at: axis)
        return result
    }

    private static func allocateStorage(
        bufferSize: Int,
        observationShape: [Int],
        observationDtype: DType,
        actionDim: Int,
        optimizeMemory: Bool
    ) -> (MLXArray, MLXArray?, MLXArray, MLXArray, MLXArray, MLXArray) {
        let obs = MLX.zeros([bufferSize] + observationShape, dtype: observationDtype)
        let nextObs =
            optimizeMemory
            ? nil : MLX.zeros([bufferSize] + observationShape, dtype: observationDtype)
        let acts = MLX.zeros([bufferSize, actionDim])
        let rews = MLX.zeros([bufferSize])
        let dns = MLX.zeros([bufferSize])
        let tos = MLX.zeros([bufferSize])

        return (obs, nextObs, acts, rews, dns, tos)
    }

    private func validateStoredObs(_ obs: MLXArray) throws {
        guard obs.shape == storageObsShape else {
            throw GymnazoError.invalidConfiguration(
                "Expected observation shape \(storageObsShape), got \(obs.shape)"
            )
        }
        guard obs.dtype == storageObsDtype else {
            throw GymnazoError.invalidConfiguration(
                "Expected observation dtype \(storageObsDtype), got \(obs.dtype)"
            )
        }
    }

    private func validateAction(_ action: MLXArray) throws {
        let flat = action.reshaped([-1])
        guard flat.size == actionDim else {
            throw GymnazoError.invalidAction(
                "Expected action dimension \(actionDim), got \(flat.size)"
            )
        }
    }

    private func sampleIndices(
        batchSize: Int,
        generator: inout SeededGenerator
    ) throws -> [Int] {
        var indices: [Int] = []
        indices.reserveCapacity(batchSize)

        if !config.optimizeMemoryUsage || !isBufferFull {
            let upper = count
            indices.append(
                contentsOf: try randomIndices(
                    batchSize: batchSize,
                    upperBound: upper,
                    generator: &generator
                ))
            return indices
        }

        let upper = bufferSize - 1
        for idx in try randomIndices(
            batchSize: batchSize,
            upperBound: upper,
            generator: &generator
        ) {
            var idx = idx
            if idx >= position { idx += 1 }
            indices.append(idx)
        }
        return indices
    }

    private func randomIndices(
        batchSize: Int,
        upperBound: Int,
        generator: inout SeededGenerator
    ) throws -> [Int] {
        precondition(upperBound > 0)
        return (0..<batchSize).map { _ in
            min(Int(generator.uniform(in: 0...1) * Float(upperBound)), upperBound - 1)
        }
    }

    private func toStoredObservation(_ obs: MLXArray) -> MLXArray {
        guard let frameStack = config.frameStack else { return obs }
        let lastIdx = frameStack.size - 1
        if resolvedStackAxis == 0 {
            return obs[lastIdx]
        }
        return obs.swappedAxes(0, resolvedStackAxis)[lastIdx]
    }

    private func previousIndex(_ index: Int) -> Int {
        (index - 1 + bufferSize) % bufferSize
    }
}

private func bufferActionDim(for actionSpace: any Space) -> Int {
    if let box = boxSpace(from: actionSpace) {
        return box.shape?.reduce(1, *) ?? 1
    }
    if actionSpace is Discrete {
        return 1
    }
    if let multiDiscrete = actionSpace as? MultiDiscrete {
        return multiDiscrete.shape?.reduce(1, *) ?? 1
    }
    if let multiBinary = actionSpace as? MultiBinary {
        return multiBinary.shape?.reduce(1, *) ?? 1
    }
    return actionSpace.shape?.reduce(1, *) ?? 1
}
