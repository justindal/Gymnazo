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

        public init(
            bufferSize: Int,
            optimizeMemoryUsage: Bool = false,
            handleTimeoutTermination: Bool = true,
            frameStack: FrameStackConfig? = nil,
            seed: UInt64? = nil
        ) {
            precondition(bufferSize > 0)
            self.bufferSize = bufferSize
            self.optimizeMemoryUsage = optimizeMemoryUsage
            self.handleTimeoutTermination = handleTimeoutTermination
            self.frameStack = frameStack
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
    private var keyState: MLXArray?
    private let storageObsShape: [Int]
    private let storageObsDtype: DType
    private let zeroFrame: MLXArray?
    private let resolvedStackAxis: Int

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

        let config = Self.config(
            config, observationSpace: observationSpace
        )
        let storageObsShape = Self.storageObservationShape(
            observationSpace: observationSpace,
            frameStack: config.frameStack
        )
        let storageObsDtype = observationSpace.dtype ?? .float32
        let zeroFrame: MLXArray?
        if config.frameStack?.padding == .zero {
            zeroFrame = MLX.zeros(storageObsShape, dtype: storageObsDtype)
            if let zeroFrame {
                eval(zeroFrame)
            }
        } else {
            zeroFrame = nil
        }

        let resolvedAxis: Int
        if let frameStack = config.frameStack,
            let obsNdim = observationSpace.shape?.count
        {
            resolvedAxis = frameStack.resolvedAxis(ndim: obsNdim)
        } else {
            resolvedAxis = 0
        }

        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        self.config = config
        self.numEnvs = numEnvs
        self.storageObsShape = storageObsShape
        self.storageObsDtype = storageObsDtype
        self.zeroFrame = zeroFrame
        self.resolvedStackAxis = resolvedAxis

        let (obs, nextObs, acts, rews, dns, tos) = Self.allocateStorage(
            bufferSize: config.bufferSize,
            observationShape: storageObsShape,
            observationDtype: storageObsDtype,
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

        self.keyState = config.seed.map { MLXRandom.key($0) }
    }

    public mutating func reset() {
        let (obs, nextObs, acts, rews, dns, tos) = Self.allocateStorage(
            bufferSize: config.bufferSize,
            observationShape: storageObsShape,
            observationDtype: storageObsDtype,
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
        keyState = config.seed.map { MLXRandom.key($0) }
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
        let storedObs = toStoredObservation(obs)
        let storedNextObs = toStoredObservation(nextObs)

        observations[position] = storedObs
        actions[position] = action.reshaped([-1])
        rewards[position] = reward
        dones[position] = MLXArray(doneValue)
        timeouts[position] = MLXArray(timeoutValue)

        if config.optimizeMemoryUsage {
            let nextPos = (position + 1) % bufferSize
            observations[nextPos] = storedNextObs
        } else {
            nextObservations![position] = storedNextObs
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
        let samplingKey: MLXArray
        if let state = keyState {
            let (sampleKey, nextKey) = MLX.split(key: state, stream: .cpu)
            keyState = nextKey
            samplingKey = sampleKey
        } else {
            samplingKey = key
        }
        let rawIndices = sampleIndices(batchSize: batchSize, key: samplingKey)
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
            (obsBatch, nextObsBatch) = sampleFrameStacks(rawIndices: rawIndices)
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

    private func sampleFrameStacks(rawIndices: [Int]) -> (MLXArray, MLXArray) {
        let frameStack = config.frameStack!
        let stackSize = frameStack.size
        let n = count
        let batchCount = rawIndices.count
        let frameShape = storageObsShape
        let isZeroPad = frameStack.padding == .zero

        let donesArray = prefetchDones(count: n)

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
                if isEpisodeStartCPU(cursor, donesArray: donesArray) {
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

    private func prefetchDones(count n: Int) -> [Float] {
        let slice = n == bufferSize ? dones : dones[0..<n]
        eval(slice)
        return slice.asArray(Float.self)
    }

    private func isEpisodeStartCPU(_ index: Int, donesArray: [Float]) -> Bool {
        if !isBufferFull {
            if index == 0 { return true }
            return donesArray[index - 1] > 0.5
        }
        if index == position { return true }
        let previous = previousIndex(index)
        return donesArray[previous] > 0.5
    }

    private static func config(
        _ config: Configuration,
        observationSpace: any Space
    ) -> Configuration {
        let obsDtype = observationSpace.dtype ?? .float32
        let storageShape = storageObservationShape(
            observationSpace: observationSpace,
            frameStack: config.frameStack
        )
        let bytesPerObs = storageShape.reduce(1, *) * obsDtype.size
        guard bytesPerObs > 0 else { return config }

        let maxBufferBytes = GPU.deviceInfo().maxBufferSize
        guard maxBufferBytes > 0 else { return config }

        let maxSafe = maxBufferBytes / bytesPerObs
        guard config.bufferSize > maxSafe else { return config }

        let bufferSize = max(1, maxSafe)
        print(
            "[ReplayBuffer] bufferSize \(config.bufferSize) would require "
                + "\(config.bufferSize * bytesPerObs) bytes per observation buffer, "
                + "exceeding Metal's \(maxBufferBytes)-byte limit on this device. "
                + "Capping to \(bufferSize)."
        )
        return Configuration(
            bufferSize: bufferSize,
            optimizeMemoryUsage: config.optimizeMemoryUsage,
            handleTimeoutTermination: config.handleTimeoutTermination,
            frameStack: config.frameStack,
            seed: config.seed
        )
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
        actionSpace: any Space,
        optimizeMemory: Bool
    ) -> (MLXArray, MLXArray?, MLXArray, MLXArray, MLXArray, MLXArray) {
        let actionDim = bufferActionDim(for: actionSpace)

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

    private func sampleIndices(batchSize: Int, key: MLXArray) -> [Int] {
        var indices: [Int] = []
        indices.reserveCapacity(batchSize)

        if !config.optimizeMemoryUsage || !isBufferFull {
            let upper = count
            indices.append(
                contentsOf: randomIndices(batchSize: batchSize, upperBound: upper, key: key))
            return indices
        }

        let upper = bufferSize - 1
        for idx in randomIndices(batchSize: batchSize, upperBound: upper, key: key) {
            var idx = idx
            if idx >= position { idx += 1 }
            indices.append(idx)
        }
        return indices
    }

    private func randomIndices(batchSize: Int, upperBound: Int, key: MLXArray) -> [Int] {
        precondition(upperBound > 0)
        let randomValues = MLX.uniform(0.0..<1.0, [batchSize], key: key, stream: .cpu)
        eval(randomValues)
        let randomList = randomValues.asArray(Float.self)
        let upperBoundF = Float(upperBound)
        return randomList.map { min(Int($0 * upperBoundF), upperBound - 1) }
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
    (actionSpace as? Box)?.shape?.reduce(1, *) ?? 1
}
