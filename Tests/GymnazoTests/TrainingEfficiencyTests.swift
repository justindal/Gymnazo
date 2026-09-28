import MLX
import MLXNN
import MLXOptimizers
import Testing

@testable import Gymnazo

@Suite("Training efficiency", .serialized)
struct TrainingEfficiencyTests {
    @Test
    func compilationUsesConcretePolicyExtractorInsteadOfOptionalModelDefault() throws {
        let policy = try DQNPolicy(
            observationSpace: Box(low: -1, high: 1, shape: [4]), nActions: 2,
            config: DQNPolicyConfig(netArch: [16], featuresExtractor: .flatten)
        )
        let imagePolicy = try DQNPolicy(
            observationSpace: Box(low: 0, high: 1, shape: [36, 36, 1]), nActions: 2,
            config: DQNPolicyConfig(netArch: [16], featuresExtractor: .natureCNN(featuresDim: 16))
        )
        #expect(supportsCompilation(policy.featuresExtractor))
        #expect(supportsCompilation(imagePolicy.featuresExtractor))
    }

    @Test
    @MainActor
    func compiledDQNStopsChangingWeightsWhenScheduleReachesZero() async throws {
        let model = try DQN(
            env: SeededDiscreteTrainingEnvironment(),
            learningRate: ScheduleFunction { $0 > 0.5 ? 0.01 : 0 },
            policyConfig: DQNPolicyConfig(netArch: [16], featuresExtractor: .flatten),
            config: DQNConfig(
                bufferSize: 32, learningStarts: 0, batchSize: 2,
                trainFrequency: .init(frequency: 1),
                explorationInitialEps: 0, explorationFinalEps: 0
            ), seed: 43
        )
        let initial = await model.policyParameterValues()
        try await model.learn(
            totalTimesteps: 8,
            callbacks: LearnCallbacks(onStep: { timestep, _, _ in timestep < 4 })
        )
        let midway = await model.policyParameterValues()
        #expect(midway != initial)
        try await model.learn(totalTimesteps: 8, callbacks: nil, resetProgress: false)
        #expect(await model.policyParameterValues() == midway)
    }

    @Test
    func compiledAdamTracksLearningRateAndMatchesEagerUpdates() {
        let eager = Linear(2, 1)
        let compiled = Linear(2, 1)
        compiled.update(parameters: eager.parameters())
        let reference = Adam(learningRate: 0.01)
        let optimizer = OptimizerConfig.adam().makeScheduled(learningRate: 0.01)
        let lossAndGrad = valueAndGrad(model: eager) { (model: Linear, x: MLXArray, y: MLXArray) in
            MLX.mean((model(x) - y) ** 2)
        }
        let compiledLossAndGrad = valueAndGrad(model: compiled) {
            (model: Linear, x: MLXArray, y: MLXArray) in
            MLX.mean((model(x) - y) ** 2)
        }
        let step = compile(inputs: [compiled, optimizer], outputs: [compiled, optimizer]) {
            (x: MLXArray) -> MLXArray in
            let (loss, gradients) = compiledLossAndGrad(compiled, x, MLXArray(Float(0)))
            optimizer.update(model: compiled, gradients: gradients)
            return loss
        }
        let x = MLXArray([Float(1), 2, 3, 4], [2, 2])
        for rate: Float in [0.01, 0.003, 0, 0.02] {
            reference.learningRate = rate
            optimizer.learningRate = rate
            let (_, gradients) = lossAndGrad(eager, x, MLXArray(Float(0)))
            reference.update(model: eager, gradients: gradients)
            let loss = step(x)
            eval(eager, reference, compiled, optimizer, loss)
            for (a, b) in zip(
                eager.parameters().flattenedValues(), compiled.parameters().flattenedValues())
            {
                #expect(MLX.allClose(a, b, atol: 1e-5).item(Bool.self))
            }
        }
    }

    @Test(arguments: [false, true])
    func replayBatchMatchesSequentialInsertionAcrossWraps(optimizeMemory: Bool) throws {
        let config = ReplayBuffer.Configuration(
            bufferSize: 5,
            optimizeMemoryUsage: optimizeMemory,
            handleTimeoutTermination: !optimizeMemory,
            seed: 7
        )
        var sequential = try ReplayBuffer(
            observationSpace: Box(low: -100, high: 100, shape: [2]),
            actionSpace: Discrete(n: 2), config: config
        )
        var batched = try ReplayBuffer(
            observationSpace: Box(low: -100, high: 100, shape: [2]),
            actionSpace: Discrete(n: 2), config: config
        )
        for count in [3, 4, 13, 1] {
            let obs = MLXArray((0..<(count * 2)).map(Float.init), [count, 2])
            let next = obs + 0.5
            let actions = MLXArray((0..<count).map { Float($0 % 2) }, [count, 1])
            let rewards = MLXArray((0..<count).map(Float.init))
            let terminations = (0..<count).map { $0 % 3 == 0 }
            let truncations = (0..<count).map { $0 % 4 == 0 }
            for index in 0..<count {
                try sequential.add(
                    obs: obs[index], action: actions[index], reward: rewards[index],
                    nextObs: next[index], terminated: terminations[index],
                    truncated: truncations[index]
                )
            }
            try batched.addBatch(
                obs: obs, actions: actions, rewards: rewards, nextObs: next,
                terminations: terminations, truncations: truncations
            )
            #expect(sequential.position == batched.position)
            #expect(sequential.isBufferFull == batched.isBufferFull)
            #expect(sequential.doneValues == batched.doneValues)
            let a = try sequential.sample(2, key: MLX.key(9))
            let b = try batched.sample(2, key: MLX.key(9))
            for (first, second) in zip(
                [
                    sequential.observations, a.obs, a.nextObs, a.actions, a.rewards, a.dones,
                    a.timeouts,
                ],
                [batched.observations, b.obs, b.nextObs, b.actions, b.rewards, b.dones, b.timeouts]
            ) {
                #expect(MLX.arrayEqual(first, second).item(Bool.self))
            }
        }
    }

    @Test
    func preparedRolloutPreservesSamplesAcrossEpochsAndPartialBatches() throws {
        var buffer = RolloutBuffer(
            bufferSize: 3, observationSpace: Box(low: -10, high: 10, shape: [1]),
            actionSpace: Discrete(n: 2), numEnvs: 2
        )
        for index in 0..<3 {
            buffer.append(
                RolloutStep(
                    observation: MLXArray([Float(index * 2), Float(index * 2 + 1)], [2, 1]),
                    action: MLXArray([Int32(0), 1]), reward: MLXArray([Float(1), 1]),
                    episodeStart: MLXArray([Float(0), 0]), value: MLXArray([Float(0), 0]),
                    logProb: MLXArray([Float(0), 0])
                ))
        }
        buffer.computeReturnsAndAdvantages(
            lastValues: MLXArray([Float(0), 0]), dones: MLXArray([Float(1), 1]),
            gamma: 1, gaeLambda: 1
        )
        let prepared = try #require(buffer.prepared())
        buffer.reset()
        var generator = SeededGenerator(seed: 3)
        for _ in 0..<2 {
            let batches = Array(prepared.batches(batchSize: 4, generator: &generator))
            #expect(batches.map { $0.values.size } == [4, 2])
            let observations = batches.flatMap { $0.observations.asArray(Float.self) }
            #expect(observations.sorted() == [0, 1, 2, 3, 4, 5])
            for batch in batches {
                let ids = batch.observations.asArray(Float.self)
                let returns = batch.returns.asArray(Float.self)
                #expect(zip(ids, returns).allSatisfy { 3 - Int($0.0) / 2 == Int($0.1) })
            }
        }
    }
    @Test(arguments: [0, -1])
    func replayBatchPreservesCompressedFrames(axis: Int) throws {
        let shape = axis == 0 ? [3, 2] : [2, 3]
        let space = Box(low: -100, high: 100, shape: shape)
        let config = ReplayBuffer.Configuration(
            bufferSize: 5, frameStack: .init(size: 3, padding: .zero, axis: axis), seed: 3
        )
        var sequential = try ReplayBuffer(
            observationSpace: space, actionSpace: Discrete(n: 2), config: config)
        var batched = try ReplayBuffer(
            observationSpace: space, actionSpace: Discrete(n: 2), config: config)
        let count = 8
        let obs = MLXArray((0..<(count * 6)).map(Float.init), [count] + shape)
        let actions = MLX.zeros([count, 1])
        let rewards = MLX.ones([count])
        let terminations = (0..<count).map { $0 == 4 }
        for index in 0..<count {
            try sequential.add(
                obs: obs[index], action: actions[index], reward: rewards[index],
                nextObs: obs[index] + 1, terminated: terminations[index], truncated: false
            )
        }
        try batched.addBatch(
            obs: obs, actions: actions, rewards: rewards, nextObs: obs + 1,
            terminations: terminations, truncations: [Bool](repeating: false, count: count)
        )
        let reference = try sequential.sample(5, key: MLX.key(3))
        let result = try batched.sample(5, key: MLX.key(3))
        #expect(MLX.arrayEqual(reference.obs, result.obs).item(Bool.self))
        #expect(MLX.arrayEqual(reference.nextObs, result.nextObs).item(Bool.self))
    }

}
