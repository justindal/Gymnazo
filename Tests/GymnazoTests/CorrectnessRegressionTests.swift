import Foundation
import MLX
import Testing

@testable import Gymnazo

@Suite("Correctness regressions", .serialized)
struct CorrectnessRegressionTests {
    @Test(arguments: [false, true])
    func replayTimeoutMasksRequirePureTruncation(batched: Bool) throws {
        var buffer = try ReplayBuffer(
            observationSpace: Box(low: 0, high: 10, shape: [1]),
            actionSpace: Discrete(n: 2),
            config: .init(bufferSize: 4, seed: 11)
        )
        let terminations = [false, false, true, true]
        let truncations = [false, true, false, true]
        if batched {
            try buffer.addBatch(
                obs: MLXArray([Float(0), 1, 2, 3], [4, 1]),
                actions: MLX.zeros([4, 1]),
                rewards: MLX.ones([4]),
                nextObs: MLXArray([Float(1), 2, 3, 4], [4, 1]),
                terminations: terminations,
                truncations: truncations
            )
        } else {
            for index in 0..<4 {
                try buffer.add(
                    obs: MLXArray([Float(index)]),
                    action: MLXArray(Int32(0)),
                    reward: MLXArray(Float(1)),
                    nextObs: MLXArray([Float(index + 1)]),
                    terminated: terminations[index],
                    truncated: truncations[index]
                )
            }
        }
        eval(buffer.dones, buffer.timeouts)
        #expect(buffer.dones.asArray(Float.self) == [0, 1, 1, 1])
        #expect(buffer.timeouts.asArray(Float.self) == [0, 1, 0, 0])

        var observed: Set<Int> = []
        for seed in 0..<64 {
            let sample = try buffer.sample(1, key: MLX.key(UInt64(seed)))
            eval(sample.obs, sample.dones)
            let index = Int(sample.obs.asArray(Float.self)[0])
            let terminal = sample.dones.asArray(Float.self)[0]
            #expect(terminal == [Float(0), 0, 1, 1][index])
            observed.insert(index)
        }
        #expect(observed == Set(0..<4))
    }

    @Test
    func optimizedCompressedReplayPadsAtOldestValidFrame() throws {
        var buffer = try ReplayBuffer(
            observationSpace: Box(low: 0, high: 10, shape: [2, 1]),
            actionSpace: Discrete(n: 2),
            config: .init(
                bufferSize: 4, optimizeMemoryUsage: true,
                handleTimeoutTermination: false,
                frameStack: .init(size: 2, padding: .reset), seed: 13
            )
        )
        for index in 0..<4 {
            try buffer.add(
                obs: MLXArray([Float(index), Float(index)], [2, 1]),
                action: MLXArray(Int32(0)), reward: MLXArray(Float(1)),
                nextObs: MLXArray([Float(index + 1), Float(index + 1)], [2, 1]),
                terminated: false, truncated: false
            )
        }
        var foundOldest = false
        for seed in 0..<32 {
            let sample = try buffer.sample(1, key: MLX.key(UInt64(seed)))
            eval(sample.obs)
            let frames = sample.obs.asArray(Float.self)
            if frames[1] == 1 {
                foundOldest = true
                #expect(frames == [1, 1])
            }
        }
        #expect(foundOldest)
    }

    @Test(arguments: [false, true])
    func replayRejectsInterleavedCompressedOrOptimizedHistory(optimized: Bool) throws {
        let config = ReplayBuffer.Configuration(
            bufferSize: 8, optimizeMemoryUsage: optimized,
            handleTimeoutTermination: !optimized,
            frameStack: optimized ? nil : .init(size: 2)
        )
        let space = Box(low: 0, high: 10, shape: optimized ? [1] : [2, 1])
        #expect(throws: GymnazoError.self) {
            _ = try ReplayBuffer(
                observationSpace: space, actionSpace: Discrete(n: 2),
                config: config, numEnvs: 2
            )
        }
    }

    @Test
    func replayRejectsCollectionWithMismatchedEnvironmentCount() throws {
        var buffer = try ReplayBuffer(
            observationSpace: Box(low: 0, high: 10, shape: [1]),
            actionSpace: Discrete(n: 2),
            config: .init(bufferSize: 8, optimizeMemoryUsage: true, handleTimeoutTermination: false)
        )
        let observation = try TensorValue([Float(0)], shape: [1])
        let reset = ValueVectorReset(
            observations: [observation, observation], infos: [[:], [:]]
        )
        let collection = ValueCollection<TensorValue<Float>, Int>(reset: reset, frames: [])
        #expect(throws: GymnazoError.self) { try buffer.add(collection) }
    }

    @Test
    func csvKeepsColumnsWhenMetricsAppearAndDisappearAcrossOutputs() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = directory.appendingPathComponent("first.csv")
        let second = directory.appendingPathComponent("second.csv")
        let logger = StandardLogger(outputs: [.csv(first), .csv(second)])

        logger.record("rollout/reward", value: 10.0)
        logger.dump(step: 1)
        logger.record("train/loss", value: 0.25)
        logger.dump(step: 2)
        logger.record("rollout/reward", value: 12.0)
        logger.dump(step: 3)

        let expected = "step,rollout/reward,train/loss\n1,10,\n2,,0.2500\n3,12,\n"
        #expect(try String(contentsOf: first, encoding: .utf8) == expected)
        #expect(try String(contentsOf: second, encoding: .utf8) == expected)
    }

    @Test
    func failedBestModelSaveRetriesAtLowerObservedValue() {
        enum SaveFailure: Error { case expected }
        var value = 100.0
        var attempts: [Double] = []
        let callback = BestModelCallback(
            savePath: URL(fileURLWithPath: "/tmp/best-model-test"),
            valueProvider: { value },
            saveAction: { _ in
                attempts.append(value)
                if attempts.count == 1 { throw SaveFailure.expected }
            },
            verbose: 0
        )
        _ = callback.onStep(locals: .init(numTimesteps: 1))
        #expect(callback.best == 100)
        value = 90
        _ = callback.onStep(locals: .init(numTimesteps: 2))
        value = 85
        _ = callback.onStep(locals: .init(numTimesteps: 3))
        value = 95
        _ = callback.onStep(locals: .init(numTimesteps: 4))
        #expect(callback.best == 100)
        #expect(attempts == [100, 90, 95])
    }

    @Test
    func evaluationRetriesSavingAfterFailureAtEqualReward() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let modelURL = directory.appendingPathComponent("nested/model.safetensors")
        let policy = try DQNPolicy(
            observationSpace: Box(low: -1, high: 1, shape: [4]), nActions: 2,
            config: DQNPolicyConfig(netArch: [16], featuresExtractor: .flatten)
        )
        let callback = EvalCallback(
            evalEnv: SeededDiscreteTrainingEnvironment(), policy: policy,
            config: .init(evalFreq: 1, nEvalEpisodes: 1, verbose: 0),
            bestModelSavePath: modelURL
        )

        _ = callback.onStep(locals: .init(numTimesteps: 1))
        #expect(callback.bestReward == 3)
        #expect(!FileManager.default.fileExists(atPath: modelURL.path))
        try FileManager.default.createDirectory(
            at: modelURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        _ = callback.onStep(locals: .init(numTimesteps: 2))
        #expect(callback.evaluationCount == 2)
        #expect(FileManager.default.fileExists(atPath: modelURL.path))
    }
}
