import MLX
import Testing

@testable import Gymnazo

@Suite("Buffer invariants", .serialized)
struct BufferTests {
    private func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let diff = MLX.abs(lhs.asType(.float32) - rhs.asType(.float32))
        eval(diff)
        return MLX.max(diff).item(Float.self)
    }

    @Test
    func replayCountsAndShapes() {
        let obsSpace = Box(
            low: MLXArray([-1.0 as Float, -1.0 as Float]),
            high: MLXArray([1.0 as Float, 1.0 as Float])
        )
        let actSpace = Discrete(n: 3)
        let config = ReplayBuffer.Configuration(
            bufferSize: 5,
            optimizeMemoryUsage: false,
            handleTimeoutTermination: true,
            seed: 11
        )
        var buffer = ReplayBuffer(
            observationSpace: obsSpace,
            actionSpace: actSpace,
            config: config
        )

        for step in 0..<8 {
            let obs = MLXArray([Float(step), Float(step) + 0.5])
            let action = MLXArray(Int32(step % 3))
            let reward = MLXArray(Float(step) * 0.1)
            let nextObs = MLXArray([Float(step + 1), Float(step + 1) + 0.5])
            buffer.add(
                obs: obs,
                action: action,
                reward: reward,
                nextObs: nextObs,
                terminated: false,
                truncated: false
            )
        }

        #expect(buffer.count == 5)
        #expect(buffer.isBufferFull)

        let sample = buffer.sample(3, key: MLX.key(123))
        #expect(sample.obs.shape == [3, 2])
        #expect(sample.actions.shape == [3, 1])
        #expect(sample.rewards.shape == [3])
        #expect(sample.nextObs.shape == [3, 2])
        #expect(sample.dones.shape == [3])
        #expect(sample.timeouts.shape == [3])
    }

    @Test
    func timeoutMaskingClearsDone() {
        let obsSpace = Box(
            low: MLXArray([-1.0 as Float]),
            high: MLXArray([1.0 as Float])
        )
        let actSpace = Discrete(n: 2)
        var buffer = ReplayBuffer(
            observationSpace: obsSpace,
            actionSpace: actSpace,
            config: ReplayBuffer.Configuration(
                bufferSize: 4,
                optimizeMemoryUsage: false,
                handleTimeoutTermination: true,
                seed: 22
            )
        )

        buffer.add(
            obs: MLXArray([0.0 as Float]),
            action: MLXArray(Int32(1)),
            reward: MLXArray(1.0 as Float),
            nextObs: MLXArray([0.5 as Float]),
            terminated: false,
            truncated: true
        )

        let sample = buffer.sample(1, key: MLX.key(9))
        eval(sample.dones, sample.timeouts)
        let doneValue = sample.dones.item(Float.self)
        let timeoutValue = sample.timeouts.item(Float.self)

        #expect(abs(doneValue - 0.0) < 1e-6)
        #expect(abs(timeoutValue - 1.0) < 1e-6)
    }

    @Test
    func frameStackCompressionRoundTrip() {
        let obsSpace = Box(low: 0, high: 255, shape: [4, 2], dtype: .uint8)
        let actSpace = Discrete(n: 2)
        var buffer = ReplayBuffer(
            observationSpace: obsSpace,
            actionSpace: actSpace,
            config: ReplayBuffer.Configuration(
                bufferSize: 8,
                optimizeMemoryUsage: true,
                handleTimeoutTermination: false,
                frameStack: .init(size: 4, padding: .zero)
            )
        )

        let zero = MLXArray([UInt8(0), UInt8(0)])
        let frame0 = MLXArray([UInt8(10), UInt8(20)])
        let frame1 = MLXArray([UInt8(11), UInt8(21)])
        let obs = MLX.stacked([zero, zero, zero, frame0], axis: 0).asType(.uint8)
        let nextObs = MLX.stacked([zero, zero, frame0, frame1], axis: 0).asType(.uint8)

        buffer.add(
            obs: obs,
            action: MLXArray(Int32(1)),
            reward: MLXArray(1.0 as Float),
            nextObs: nextObs,
            terminated: false,
            truncated: false
        )

        #expect(buffer.observations.shape == [8, 2])

        let sample = buffer.sample(1, key: MLX.key(5))
        #expect(sample.obs.shape == [1, 4, 2])
        #expect(sample.nextObs.shape == [1, 4, 2])

        let sampledObs = sample.obs[0]
        let sampledNextObs = sample.nextObs[0]

        #expect(maxAbsDiff(sampledObs, obs) < 1e-6)
        #expect(maxAbsDiff(sampledNextObs, nextObs) < 1e-6)
    }

    @Test
    func frameStackChannelLastShapes() {
        let obsSpace = Box(low: 0, high: 255, shape: [84, 84, 4], dtype: .uint8)
        let actSpace = Discrete(n: 5)
        var buffer = ReplayBuffer(
            observationSpace: obsSpace,
            actionSpace: actSpace,
            config: ReplayBuffer.Configuration(
                bufferSize: 16,
                optimizeMemoryUsage: true,
                handleTimeoutTermination: false,
                frameStack: .init(size: 4, padding: .zero, axis: -1)
            )
        )

        for step in 0..<8 {
            let frame = MLXArray(
                Array(repeating: UInt8(step), count: 84 * 84 * 4),
                [84, 84, 4]
            )
            let nextFrame = MLXArray(
                Array(repeating: UInt8(step + 1), count: 84 * 84 * 4),
                [84, 84, 4]
            )
            buffer.add(
                obs: frame,
                action: MLXArray(Int32(step % 5)),
                reward: MLXArray(Float(step) * 0.1),
                nextObs: nextFrame,
                terminated: step == 4,
                truncated: false
            )
        }

        #expect(buffer.observations.shape == [16, 84, 84])

        let sample = buffer.sample(4, key: MLX.key(42))
        #expect(sample.obs.shape == [4, 84, 84, 4])
        #expect(sample.nextObs.shape == [4, 84, 84, 4])
        #expect(sample.actions.shape == [4, 1])
        #expect(sample.rewards.shape == [4])
        #expect(sample.dones.shape == [4])
    }

    @Test
    func frameStackChannelLastValues() {
        let stackSize = 4
        let obsSpace = Box(low: 0, high: 255, shape: [2, 2, stackSize], dtype: .uint8)
        let actSpace = Discrete(n: 2)
        var buffer = ReplayBuffer(
            observationSpace: obsSpace,
            actionSpace: actSpace,
            config: ReplayBuffer.Configuration(
                bufferSize: 16,
                optimizeMemoryUsage: true,
                handleTimeoutTermination: false,
                frameStack: .init(size: stackSize, padding: .zero, axis: -1)
            )
        )

        for step in 0..<6 {
            let val = UInt8(step + 1)
            var frames: [[UInt8]] = []
            for ch in 0..<stackSize {
                let frameVal: UInt8 =
                    ch < stackSize - 1 ? UInt8(max(0, Int(val) - (stackSize - 1 - ch))) : val
                frames.append(Array(repeating: frameVal, count: 4))
            }
            let obsData = (0..<4).flatMap { pixel in frames.map { $0[pixel] } }
            let obs = MLXArray(obsData, [2, 2, stackSize]).asType(.uint8)

            let nextVal = UInt8(step + 2)
            var nextFrames: [[UInt8]] = []
            for ch in 0..<stackSize {
                let frameVal: UInt8 =
                    ch < stackSize - 1
                    ? UInt8(max(0, Int(nextVal) - (stackSize - 1 - ch))) : nextVal
                nextFrames.append(Array(repeating: frameVal, count: 4))
            }
            let nextObsData = (0..<4).flatMap { pixel in nextFrames.map { $0[pixel] } }
            let nextObs = MLXArray(nextObsData, [2, 2, stackSize]).asType(.uint8)

            buffer.add(
                obs: obs,
                action: MLXArray(Int32(0)),
                reward: MLXArray(1.0 as Float),
                nextObs: nextObs,
                terminated: false,
                truncated: false
            )
        }

        let sample = buffer.sample(1, key: MLX.key(99))
        #expect(sample.obs.shape == [1, 2, 2, stackSize])
        #expect(sample.nextObs.shape == [1, 2, 2, stackSize])
    }

    @Test
    func frameStackAxis0Shapes() {
        let obsSpace = Box(low: 0, high: 255, shape: [4, 6], dtype: .uint8)
        let actSpace = Discrete(n: 3)
        var buffer = ReplayBuffer(
            observationSpace: obsSpace,
            actionSpace: actSpace,
            config: ReplayBuffer.Configuration(
                bufferSize: 10,
                optimizeMemoryUsage: true,
                handleTimeoutTermination: false,
                frameStack: .init(size: 4, padding: .reset, axis: 0)
            )
        )

        for step in 0..<6 {
            let obs = MLXArray(
                Array(repeating: UInt8(step), count: 4 * 6), [4, 6]
            ).asType(.uint8)
            let nextObs = MLXArray(
                Array(repeating: UInt8(step + 1), count: 4 * 6), [4, 6]
            ).asType(.uint8)
            buffer.add(
                obs: obs,
                action: MLXArray(Int32(step % 3)),
                reward: MLXArray(Float(step)),
                nextObs: nextObs,
                terminated: false,
                truncated: false
            )
        }

        #expect(buffer.observations.shape == [10, 6])

        let sample = buffer.sample(2, key: MLX.key(7))
        #expect(sample.obs.shape == [2, 4, 6])
        #expect(sample.nextObs.shape == [2, 4, 6])
    }

    @Test
    func frameStackChannelLastZeroPadding() {
        let stackSize = 3
        let obsSpace = Box(low: 0, high: 255, shape: [4, 4, stackSize], dtype: .uint8)
        let actSpace = Discrete(n: 2)
        var buffer = ReplayBuffer(
            observationSpace: obsSpace,
            actionSpace: actSpace,
            config: ReplayBuffer.Configuration(
                bufferSize: 10,
                optimizeMemoryUsage: true,
                handleTimeoutTermination: false,
                frameStack: .init(size: stackSize, padding: .zero, axis: -1)
            )
        )

        let zeroChannel = MLXArray.zeros([4, 4, 1]).asType(.uint8)
        let fiveChannel = MLXArray.full([4, 4, 1], values: MLXArray(UInt8(5))).asType(.uint8)
        let sixChannel = MLXArray.full([4, 4, 1], values: MLXArray(UInt8(6))).asType(.uint8)

        let obs = MLX.concatenated(
            [zeroChannel, zeroChannel, fiveChannel], axis: 2
        )
        let nextObs = MLX.concatenated(
            [zeroChannel, fiveChannel, sixChannel], axis: 2
        )
        eval(obs, nextObs)

        buffer.add(
            obs: obs,
            action: MLXArray(Int32(1)),
            reward: MLXArray(1.0 as Float),
            nextObs: nextObs,
            terminated: false,
            truncated: false
        )

        let sample = buffer.sample(1, key: MLX.key(3))
        #expect(sample.obs.shape == [1, 4, 4, stackSize])
        #expect(sample.nextObs.shape == [1, 4, 4, stackSize])

        eval(sample.obs)
        let obsSlice = sample.obs[0]
        eval(obsSlice)
        let zeroChannels = obsSlice[0..., 0..., ..<(stackSize - 1)]
        eval(zeroChannels)
        let maxVal = MLX.max(zeroChannels).item(Float.self)
        #expect(maxVal < 1e-6)

        let lastChannel = obsSlice[0..., 0..., (stackSize - 1)..<stackSize]
        eval(lastChannel)
        let minVal = MLX.min(lastChannel).item(Float.self)
        #expect(minVal > 4.0)
    }

    @Test
    func frameStackEpisodeBoundary() {
        let stackSize = 3
        let obsSpace = Box(low: 0, high: 255, shape: [stackSize, 2], dtype: .uint8)
        let actSpace = Discrete(n: 2)
        var buffer = ReplayBuffer(
            observationSpace: obsSpace,
            actionSpace: actSpace,
            config: ReplayBuffer.Configuration(
                bufferSize: 10,
                optimizeMemoryUsage: true,
                handleTimeoutTermination: false,
                frameStack: .init(size: stackSize, padding: .zero, axis: 0)
            )
        )

        for step in 0..<3 {
            let obs = MLXArray(
                Array(repeating: UInt8(step + 1), count: stackSize * 2), [stackSize, 2]
            ).asType(.uint8)
            let nextObs = MLXArray(
                Array(repeating: UInt8(step + 2), count: stackSize * 2), [stackSize, 2]
            ).asType(.uint8)
            buffer.add(
                obs: obs,
                action: MLXArray(Int32(0)),
                reward: MLXArray(1.0 as Float),
                nextObs: nextObs,
                terminated: step == 1,
                truncated: false
            )
        }

        let sample = buffer.sample(3, key: MLX.key(55))
        #expect(sample.obs.shape == [3, stackSize, 2])
        #expect(sample.nextObs.shape == [3, stackSize, 2])
    }

    @Test
    func rolloutFiniteStatsAndBatches() {
        let obsSpace = Box(low: -1.0, high: 1.0, shape: [3])
        let actSpace = Box(low: -1.0, high: 1.0, shape: [1])
        var buffer = RolloutBuffer(
            bufferSize: 4,
            observationSpace: obsSpace,
            actionSpace: actSpace,
            numEnvs: 1
        )

        for index in 0..<4 {
            buffer.append(
                RolloutStep(
                    observation: MLXArray([Float(index), Float(index) + 0.1, Float(index) + 0.2]),
                    action: MLXArray([Float(index) * 0.05]),
                    reward: MLXArray(Float(index) * 0.2),
                    episodeStart: MLXArray(index == 0 ? 1.0 as Float : 0.0 as Float),
                    value: MLXArray(Float(index) * 0.1),
                    logProb: MLXArray(-0.1 as Float)
                )
            )
        }

        buffer.computeReturnsAndAdvantages(
            lastValues: MLXArray(0.25 as Float),
            dones: MLXArray(0.0 as Float),
            gamma: 0.99,
            gaeLambda: 0.95
        )

        let stats = buffer.valuesAndReturns()
        #expect(stats.values.count == 4)
        #expect(stats.returns.count == 4)
        #expect(!stats.values.contains(where: { !$0.isFinite }))
        #expect(!stats.returns.contains(where: { !$0.isFinite }))

        let batches = buffer.batches(batchSize: 2, key: MLX.key(77))
        #expect(batches.count == 2)

        for batch in batches {
            #expect(batch.observations.shape == [2, 3])
            #expect(batch.actions.shape == [2, 1])
            #expect(batch.values.shape == [2])
            #expect(batch.logProbs.shape == [2])
            #expect(batch.advantages.shape == [2])
            #expect(batch.returns.shape == [2])
        }
    }
}
