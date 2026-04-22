import MLX
import MLXNN
import Testing

@testable import Gymnazo

@Suite("SAC")
struct SACTests {
    private func observationSpace() -> Box {
        Box(low: -1.0, high: 1.0, shape: [3])
    }

    private func actionSpace() -> Box {
        Box(low: -2.0, high: 2.0, shape: [1])
    }

    private func maxParameterDiff(_ lhs: ModuleParameters, _ rhs: ModuleParameters)
        -> Float
    {
        let left = Dictionary(uniqueKeysWithValues: lhs.flattened())
        let right = Dictionary(uniqueKeysWithValues: rhs.flattened())
        guard left.count == right.count else { return .infinity }

        var maxDiff: Float = 0
        for (key, leftValue) in left {
            guard let rightValue = right[key] else { return .infinity }
            let diff = MLX.max(MLX.abs(leftValue - rightValue))
            eval(diff)
            maxDiff = max(maxDiff, diff.item(Float.self))
        }
        return maxDiff
    }

    private func parameterL1Norm(_ params: ModuleParameters) -> Float {
        let flattened = Dictionary(uniqueKeysWithValues: params.flattened())
        var total: Float = 0
        for (_, value) in flattened {
            let norm = MLX.sum(MLX.abs(value))
            eval(norm)
            total += norm.item(Float.self)
        }
        return total
    }

    @Test
    @MainActor
    func targetEntropyDefaultsToNegativeActionDim() async throws {
        let sac = try SAC(
            observationSpace: observationSpace(),
            actionSpace: actionSpace(),
            networksConfig: SACNetworksConfig(
                actor: SACActorConfig(featuresExtractor: .flatten),
                critic: SACCriticConfig(shareFeaturesExtractor: false)
            ),
            seed: 1
        )
        #expect(abs((await sac.targetEntropy) + 1.0) < 1e-6)
    }

    @Test
    func policyFeatureExtractorSharingAndTargetSync() throws {
        let shared = try SACNetworks(
            observationSpace: observationSpace(),
            actionSpace: actionSpace(),
            config: SACNetworksConfig(
                actor: SACActorConfig(featuresExtractor: .flatten),
                critic: SACCriticConfig(shareFeaturesExtractor: true)
            )
        )

        guard
            let sharedCriticExtractor = shared.critic.featuresExtractor,
            let sharedCriticTargetExtractor = shared.criticTarget.featuresExtractor
        else {
            Issue.record("SAC critics should have feature extractors.")
            return
        }

        let sharedActorExtractorID = ObjectIdentifier(shared.actor.featuresExtractor as AnyObject)
        let sharedCriticExtractorID = ObjectIdentifier(sharedCriticExtractor as AnyObject)
        let sharedCriticTargetExtractorID = ObjectIdentifier(
            sharedCriticTargetExtractor as AnyObject)

        #expect(sharedActorExtractorID == sharedCriticExtractorID)
        #expect(sharedActorExtractorID != sharedCriticTargetExtractorID)

        let criticDiff = maxParameterDiff(
            shared.critic.parameters(),
            shared.criticTarget.parameters()
        )
        #expect(criticDiff < 1e-6)
    }

    @Test
    @MainActor
    func targetUpdateIntervalMatchesSB3Behavior() async throws {
        let env = try await Gymnazo.make("Pendulum")
        let sac = try SAC(
            env: env,
            networksConfig: SACNetworksConfig(
                actor: SACActorConfig(featuresExtractor: .flatten),
                critic: SACCriticConfig(shareFeaturesExtractor: false)
            ),
            config: OffPolicyConfig(
                bufferSize: 64,
                learningStarts: 0,
                batchSize: 1,
                tau: 0.005,
                gamma: 0.99,
                trainFrequency: TrainFrequency(frequency: 1, unit: .step),
                gradientSteps: .fixed(1),
                targetUpdateInterval: 10,
                optimizeMemoryUsage: false,
                handleTimeoutTermination: true,
                useSDEAtWarmup: false,
                sdeSampleFreq: -1,
                sdeSupported: true
            ),
            seed: 42
        )

        let targetBeforeNorm = parameterL1Norm((await sac.criticTarget).parameters())
        try await sac.learn(totalTimesteps: 3, callbacks: nil)
        let targetAfterNorm = parameterL1Norm((await sac.criticTarget).parameters())

        #expect(abs(targetAfterNorm - targetBeforeNorm) > 1e-8)
    }

    @Test
    @MainActor
    func learnSmoke() async throws {
        let trainingEnv = try await Gymnazo.make("Pendulum")
        let sac = try SAC(
            env: trainingEnv,
            networksConfig: SACNetworksConfig(
                actor: SACActorConfig(featuresExtractor: .flatten),
                critic: SACCriticConfig(shareFeaturesExtractor: false)
            ),
            config: OffPolicyConfig(
                bufferSize: 256,
                learningStarts: 5,
                batchSize: 8,
                tau: 0.005,
                gamma: 0.99,
                trainFrequency: TrainFrequency(frequency: 1, unit: .step),
                gradientSteps: .fixed(1),
                targetUpdateInterval: 1,
                optimizeMemoryUsage: false,
                handleTimeoutTermination: true,
                useSDEAtWarmup: false,
                sdeSampleFreq: -1,
                sdeSupported: true
            ),
            seed: 123
        )

        try await sac.learn(totalTimesteps: 20, callbacks: nil)
        #expect(await sac.numTimesteps == 20)
    }
}
