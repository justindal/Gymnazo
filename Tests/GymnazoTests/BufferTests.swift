import MLX
import Testing

@testable import Gymnazo

@Suite("Buffer invariants", .serialized)
struct BufferTests {
    @Test
    func replayBufferCapsCountAndSampleShapes() {
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
    func replayBufferTimeoutMaskingSetsDoneToZero() {
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
    func rolloutBufferAdvantagesAndBatchesAreFinite() {
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
