import MLX
import Testing

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

private func maxAbsDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
    guard lhs.count == rhs.count else { return .infinity }
    return zip(lhs, rhs).reduce(0) { maximum, values in
        max(maximum, abs(values.0 - values.1))
    }
}

private struct TruncatingPPOEnv: Env {
    let observationSpace: any Space = Box(low: -1, high: 1, shape: [4])
    let actionSpace: any Space = Discrete(n: 2)
    var spec: EnvSpec?
    var renderMode: RenderMode?
    private var steps = 0

    mutating func step(_ action: MLXArray) throws -> Step {
        steps += 1
        let obs = MLXArray([Float(steps), 0, 0, 0])
        return Step(obs: obs, reward: 1.0, terminated: false, truncated: true)
    }

    mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        steps = 0
        return Reset(obs: MLXArray([Float(0), 0, 0, 0]))
    }
}

@Suite("PPO", .serialized)
struct PPOTests {
    @Test
    @MainActor
    func seededNetworksMatchAndDifferentSeedsDiffer() async throws {
        let policyConfig = PPOPolicyConfig(
            netArch: .shared([16]),
            featuresExtractor: .flatten,
            orthoInit: true
        )
        let first = try PPO(
            observationSpace: Box(low: -1, high: 1, shape: [4]),
            actionSpace: Discrete(n: 2),
            policyConfig: policyConfig,
            seed: 101
        )
        let second = try PPO(
            observationSpace: Box(low: -1, high: 1, shape: [4]),
            actionSpace: Discrete(n: 2),
            policyConfig: policyConfig,
            seed: 101
        )
        let different = try PPO(
            observationSpace: Box(low: -1, high: 1, shape: [4]),
            actionSpace: Discrete(n: 2),
            policyConfig: policyConfig,
            seed: 102
        )

        #expect(await first.policyParameterValues() == second.policyParameterValues())
        #expect(await first.policyParameterValues() != different.policyParameterValues())
    }

    @Test
    @MainActor
    func seededLearningProducesMatchingWeights() async throws {
        let policyConfig = PPOPolicyConfig(
            netArch: .shared([16]),
            featuresExtractor: .flatten,
            orthoInit: true
        )
        let config = PPOConfig(nSteps: 4, batchSize: 2, nEpochs: 1)
        let first = try PPO(
            env: SeededDiscreteTrainingEnvironment(),
            policyConfig: policyConfig,
            config: config,
            seed: 43
        )
        let second = try PPO(
            env: SeededDiscreteTrainingEnvironment(),
            policyConfig: policyConfig,
            config: config,
            seed: 43
        )

        try await first.learn(totalTimesteps: 12, callbacks: nil)
        try await second.learn(totalTimesteps: 12, callbacks: nil)

        let firstValues = await first.policyParameterValues()
        let secondValues = await second.policyParameterValues()
        #expect(maxAbsDifference(firstValues, secondValues) < 1e-5)
    }

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
            env: env,
            learningRate: ConstantLearningRate(3e-4),
            policyConfig: policyConfig,
            config: config,
            seed: 42
        )

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
            Issue.record(
                "Expected PPO train metrics to include loss, approx_kl, and clip_fraction.")
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
            env: env,
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

        try await ppo.learn(totalTimesteps: 32, callbacks: nil as LearnCallbacks?)
        try await ppo.evaluate(
            episodes: 1, deterministic: true, callbacks: nil as EvaluateCallbacks?)

        var evalEnv = try await Gymnazo.make("CartPole")
        let observationValues = try evalEnv.reset(seed: 77).obs
            .reshaped([-1])
            .asArray(Float.self)
        let actionValues = await ppo.actionValues(
            observation: observationValues,
            deterministic: true
        )
        #expect(!actionValues.isEmpty)
    }

    @Test
    @MainActor
    func learnBootstrapsTruncatedStepsWithoutFinalObservation() async throws {
        let env = TruncatingPPOEnv()
        let ppo = try PPO(
            env: env,
            policyConfig: PPOPolicyConfig(
                netArch: .shared([32, 32]),
                featuresExtractor: .flatten,
                activation: .tanh,
                normalizeImages: true,
                shareFeaturesExtractor: true,
                orthoInit: false
            ),
            config: PPOConfig(
                nSteps: 8,
                batchSize: 4,
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
            seed: 11
        )

        try await ppo.learn(totalTimesteps: 16, callbacks: nil as LearnCallbacks?)

        #expect(await ppo.numTimesteps == 16)
        #expect(await ppo.nUpdates > 0)
    }
}
