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

@Suite("DQN", .serialized)
struct DQNTests {
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
