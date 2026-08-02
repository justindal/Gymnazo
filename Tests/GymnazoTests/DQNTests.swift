import MLX
import Testing

@testable import Gymnazo

private actor DQNTrainMetricCollector {
    private var values: [[String: Double]] = []

    func append(_ metrics: [String: Double]) {
        values.append(metrics)
    }

    func snapshot() -> [[String: Double]] {
        values
    }
}

private actor LearningGate {
    private var entered = false
    private var entryContinuation: CheckedContinuation<Void, Never>?
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    func waitForEntry() async {
        if entered { return }
        await withCheckedContinuation { entryContinuation = $0 }
    }

    func pause() async {
        entered = true
        entryContinuation?.resume()
        entryContinuation = nil
        await withCheckedContinuation { releaseContinuation = $0 }
    }

    func release() {
        releaseContinuation?.resume()
        releaseContinuation = nil
    }
}

@Suite("DQN", .serialized)
struct DQNTests {
    @Test
    @MainActor
    func seededNetworksMatchAndDifferentSeedsDiffer() async throws {
        let config = DQNPolicyConfig(netArch: [16], featuresExtractor: .flatten)
        let first = try DQN(
            observationSpace: Box(low: -1, high: 1, shape: [4]),
            actionSpace: Discrete(n: 2),
            policyConfig: config,
            seed: 101
        )
        let second = try DQN(
            observationSpace: Box(low: -1, high: 1, shape: [4]),
            actionSpace: Discrete(n: 2),
            policyConfig: config,
            seed: 101
        )
        let different = try DQN(
            observationSpace: Box(low: -1, high: 1, shape: [4]),
            actionSpace: Discrete(n: 2),
            policyConfig: config,
            seed: 102
        )

        #expect(await first.policyParameterValues() == second.policyParameterValues())
        #expect(await first.policyParameterValues() != different.policyParameterValues())
    }

    @Test
    @MainActor
    func seededLearningProducesMatchingWeights() async throws {
        let config = DQNConfig(
            bufferSize: 32,
            learningStarts: 0,
            batchSize: 2,
            trainFrequency: .init(frequency: 1),
            gradientSteps: .fixed(1),
            explorationInitialEps: 0,
            explorationFinalEps: 0
        )
        let policyConfig = DQNPolicyConfig(netArch: [16], featuresExtractor: .flatten)
        let first = try DQN(
            env: SeededDiscreteTrainingEnvironment(),
            policyConfig: policyConfig,
            config: config,
            seed: 43
        )
        let second = try DQN(
            env: SeededDiscreteTrainingEnvironment(),
            policyConfig: policyConfig,
            config: config,
            seed: 43
        )

        try await first.learn(totalTimesteps: 12, callbacks: nil)
        try await second.learn(totalTimesteps: 12, callbacks: nil)

        #expect(await first.policyParameterValues() == second.policyParameterValues())
    }

    @Test
    @MainActor
    func concurrentSessionIsRejected() async throws {
        let env = try await Gymnazo.make("CartPole")
        let dqn = try DQN(
            env: env,
            config: DQNConfig(
                bufferSize: 8,
                learningStarts: 8,
                batchSize: 1,
                trainFrequency: .init(frequency: 1),
                gradientSteps: .fixed(1)
            ),
            seed: 17
        )
        let gate = LearningGate()
        let callbacks = LearnCallbacks(onStep: { _, _, _ in
            await gate.pause()
            return true
        })
        let training = Task {
            try await dqn.learn(totalTimesteps: 1, callbacks: callbacks)
        }

        await gate.waitForEntry()
        await #expect(throws: GymnazoError.self) {
            try await dqn.evaluate(episodes: 1)
        }
        await gate.release()
        try await training.value
    }

    @Test
    @MainActor
    func learnEmitsFiniteMetricsAndGradientSteps() async throws {
        let env = try await Gymnazo.make("CartPole")
        let config = DQNConfig(
            bufferSize: 256,
            learningStarts: 0,
            batchSize: 8,
            tau: 1.0,
            gamma: 0.99,
            trainFrequency: TrainFrequency(frequency: 1, unit: .step),
            gradientSteps: .fixed(1),
            targetUpdateInterval: 5,
            explorationFraction: 0.5,
            explorationInitialEps: 1.0,
            explorationFinalEps: 0.1,
            maxGradNorm: 10.0,
            optimizeMemoryUsage: false,
            handleTimeoutTermination: true
        )
        let policyConfig = DQNPolicyConfig(
            netArch: [32, 32],
            featuresExtractor: .flatten,
            activation: .relu,
            normalizeImages: true
        )
        let dqn = try DQN(
            env: env,
            learningRate: ConstantLearningRate(1e-4),
            policyConfig: policyConfig,
            config: config,
            optimizerConfig: DQNOptimizerConfig(),
            seed: 42
        )

        let collector = DQNTrainMetricCollector()
        let callbacks = LearnCallbacks(
            onTrain: { metrics in
                await collector.append(metrics)
            }
        )

        try await dqn.learn(totalTimesteps: 64, callbacks: callbacks)

        #expect(await dqn.numTimesteps == 64)
        #expect(await dqn.gradientSteps > 0)

        let metrics = await collector.snapshot()
        #expect(!metrics.isEmpty)

        guard let latest = metrics.last else {
            Issue.record("Expected at least one DQN train metric payload.")
            return
        }
        guard
            let loss = latest["loss"],
            let tdError = latest["tdError"],
            let meanQValue = latest["meanQValue"],
            let learningRate = latest["learningRate"]
        else {
            Issue.record(
                "Expected DQN train metrics to include loss, tdError, meanQValue, and learningRate."
            )
            return
        }

        #expect(loss.isFinite)
        #expect(tdError.isFinite)
        #expect(meanQValue.isFinite)
        #expect(learningRate.isFinite)
    }

    @Test
    @MainActor
    func epsilonDecaysTowardFinalValue() async throws {
        let env = try await Gymnazo.make("CartPole")
        let config = DQNConfig(
            bufferSize: 128,
            learningStarts: 0,
            batchSize: 8,
            tau: 1.0,
            gamma: 0.99,
            trainFrequency: TrainFrequency(frequency: 1, unit: .step),
            gradientSteps: .fixed(1),
            targetUpdateInterval: 5,
            explorationFraction: 0.5,
            explorationInitialEps: 1.0,
            explorationFinalEps: 0.2,
            maxGradNorm: 10.0,
            optimizeMemoryUsage: false,
            handleTimeoutTermination: true
        )
        let dqn = try DQN(
            env: env,
            config: config,
            seed: 5
        )

        let initialEpsilon = await dqn.epsilon
        try await dqn.learn(totalTimesteps: 40, callbacks: nil)
        let finalEpsilon = await dqn.epsilon

        #expect(finalEpsilon < initialEpsilon)
        #expect(abs(finalEpsilon - config.explorationFinalEps) < 1e-6)
    }
}
