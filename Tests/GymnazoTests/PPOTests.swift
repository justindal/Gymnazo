import Testing
import MLX
@testable import Gymnazo

private actor PPOTrainMetricCollector {
    private var values: [[String: Double]] = []

    func append(_ metrics: [String: Double]) {
        values.append(metrics)
    }

    func snapshot() -> [[String: Double]] {
        values
    }
}

@Suite("PPO", .serialized)
struct PPOTests {
    @Test
    @MainActor
    func learnEmitsFiniteMetrics() async throws {
        let env = try await Gymnazo.make("CartPole")
        let policyConfig = PPOPolicyConfig(
            netArch: .shared([32, 32]),
            featuresExtractor: .flatten,
            activation: .tanh,
            normalizeImages: true,
            shareFeaturesExtractor: true,
            orthoInit: false
        )
        let config = PPOConfig(
            nSteps: 32,
            batchSize: 8,
            nEpochs: 2,
            gamma: 0.99,
            gaeLambda: 0.95,
            clipRange: 0.2,
            clipRangeVf: nil,
            normalizeAdvantage: true,
            entCoef: 0.0,
            vfCoef: 0.5,
            maxGradNorm: 0.5,
            targetKL: nil,
            useSDE: false,
            sdeSampleFreq: -1
        )
        let ppo = try PPO(
            observationSpace: env.observationSpace,
            actionSpace: env.actionSpace,
            learningRate: ConstantLearningRate(3e-4),
            policyConfig: policyConfig,
            config: config,
            seed: 42
        )
        ppo.setEnv(env)

        let collector = PPOTrainMetricCollector()
        let callbacks = LearnCallbacks(
            onTrain: { metrics in
                await collector.append(metrics)
            }
        )

        try await ppo.learn(totalTimesteps: 64, callbacks: callbacks)
        #expect(await ppo.numTimesteps == 64)
        #expect(await ppo.nUpdates > 0)

        let metrics = await collector.snapshot()
        #expect(!metrics.isEmpty)

        guard let latest = metrics.last else {
            Issue.record("Expected at least one train metric payload.")
            return
        }
        guard
            let loss = latest[LogKey.Train.loss],
            let approxKL = latest[LogKey.Train.approxKL],
            let clipFraction = latest[LogKey.Train.clipFraction]
        else {
            Issue.record("Expected PPO train metrics to include loss, approx_kl, and clip_fraction.")
            return
        }

        #expect(loss.isFinite)
        #expect(approxKL.isFinite)
        #expect(clipFraction.isFinite)
        #expect(clipFraction >= 0.0)
        #expect(clipFraction <= 1.0)
    }

    @Test
    @MainActor
    func learnAndEvaluateSmoke() async throws {
        let env = try await Gymnazo.make("CartPole")
        let ppo = try PPO(
            observationSpace: env.observationSpace,
            actionSpace: env.actionSpace,
            policyConfig: PPOPolicyConfig(
                netArch: .shared([32, 32]),
                featuresExtractor: .flatten,
                activation: .tanh,
                normalizeImages: true,
                shareFeaturesExtractor: true,
                orthoInit: false
            ),
            config: PPOConfig(
                nSteps: 16,
                batchSize: 8,
                nEpochs: 1,
                gamma: 0.99,
                gaeLambda: 0.95,
                clipRange: 0.2,
                clipRangeVf: nil,
                normalizeAdvantage: true,
                entCoef: 0.0,
                vfCoef: 0.5,
                maxGradNorm: 0.5,
                targetKL: nil,
                useSDE: false,
                sdeSampleFreq: -1
            ),
            seed: 7
        )
        ppo.setEnv(env)

        try await ppo.learn(totalTimesteps: 32, callbacks: nil as LearnCallbacks?)
        try await ppo.evaluate(episodes: 1, deterministic: true, callbacks: nil as EvaluateCallbacks?)

        var evalEnv = try await Gymnazo.make("CartPole")
        let observation = try evalEnv.reset(seed: 77).obs
        let action = await ppo(observation: observation, deterministic: true)
        eval(action)
        #expect(action.size > 0)
    }
}
