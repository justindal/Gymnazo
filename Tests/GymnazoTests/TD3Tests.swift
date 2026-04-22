import MLX
import MLXNN
import Testing

@testable import Gymnazo

@Suite("TD3")
struct TD3Tests {
    private func observationSpace() -> Box {
        Box(low: -1.0, high: 1.0, shape: [3])
    }

    private func actionSpace() -> Box {
        Box(low: -2.0, high: 2.0, shape: [1])
    }

    private func maxParameterDiff(_ lhs: ModuleParameters, _ rhs: ModuleParameters) -> Float {
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

    @Test
    func algorithmConfigDefaultsAndLegacyNoiseMapping() {
        let defaults = TD3AlgorithmConfig()
        #expect(defaults.policyDelay == 2)
        #expect(abs(defaults.targetPolicyNoise - 0.2) < 1e-6)
        #expect(abs(defaults.targetNoiseClip - 0.5) < 1e-6)
        #expect(defaults.actionNoise == nil)

        let legacy = TD3AlgorithmConfig(actionNoiseStd: 0.15)
        guard case .some(.normal(let legacyStd)) = legacy.actionNoise else {
            Issue.record("Legacy actionNoiseStd should map to .normal noise.")
            return
        }
        #expect(abs(legacyStd - 0.15) < 1e-6)

        let sanitized = TD3AlgorithmConfig(
            policyDelay: 0,
            targetPolicyNoise: -1.0,
            targetNoiseClip: -2.0,
            actionNoise: .ornsteinUhlenbeck(
                std: -0.2,
                theta: -0.5,
                dt: 0.0,
                initialNoise: 0.25
            )
        )
        #expect(sanitized.policyDelay == 1)
        #expect(sanitized.targetPolicyNoise == 0.0)
        #expect(sanitized.targetNoiseClip == 0.0)

        guard case
            .some(
                .ornsteinUhlenbeck(
                    std: let std,
                    theta: let theta,
                    dt: let dt,
                    initialNoise: let initialNoise
                )
            ) = sanitized.actionNoise
        else {
            Issue.record("Expected sanitized OU action noise config.")
            return
        }
        #expect(std == 0.0)
        #expect(theta == 0.0)
        #expect(dt == 1e-9)
        #expect(abs(initialNoise - 0.25) < 1e-6)
    }

    @Test
    func td3SanitizesUnsupportedOffPolicyFields() throws {
        let provided = OffPolicyConfig(
            bufferSize: 1234,
            learningStarts: 5,
            batchSize: 16,
            tau: 0.01,
            gamma: 0.97,
            trainFrequency: TrainFrequency(frequency: 2, unit: .step),
            gradientSteps: .fixed(2),
            targetUpdateInterval: 7,
            optimizeMemoryUsage: false,
            handleTimeoutTermination: true,
            useSDEAtWarmup: true,
            sdeSampleFreq: 4,
            sdeSupported: true
        )

        let td3 = try TD3(
            observationSpace: observationSpace(),
            actionSpace: actionSpace(),
            config: provided,
            seed: 42
        )
        let resolved = td3.offPolicyConfig

        #expect(resolved.bufferSize == 1234)
        #expect(resolved.learningStarts == 5)
        #expect(resolved.batchSize == 16)
        #expect(abs(resolved.tau - 0.01) < 1e-12)
        #expect(abs(resolved.gamma - 0.97) < 1e-12)
        #expect(resolved.targetUpdateInterval == 1)
        #expect(resolved.useSDEAtWarmup == false)
        #expect(resolved.sdeSampleFreq == -1)
        #expect(resolved.sdeSupported == false)
    }

    @Test
    func policyFeatureExtractorSharingAndTargetSync() throws {
        let shared = try TD3Policy(
            observationSpace: observationSpace(),
            actionSpace: actionSpace(),
            featuresExtractor: .flatten,
            shareFeaturesExtractor: true
        )

        guard
            let sharedCriticExtractor = shared.critic.featuresExtractor,
            let sharedCriticTargetExtractor = shared.criticTarget.featuresExtractor
        else {
            Issue.record("TD3Policy critics should have feature extractors.")
            return
        }

        let sharedActorExtractorID = ObjectIdentifier(shared.actor.featuresExtractor as AnyObject)
        let sharedCriticExtractorID = ObjectIdentifier(sharedCriticExtractor as AnyObject)
        let sharedActorTargetExtractorID = ObjectIdentifier(shared.actorTarget.featuresExtractor as AnyObject)
        let sharedCriticTargetExtractorID = ObjectIdentifier(sharedCriticTargetExtractor as AnyObject)

        #expect(sharedActorExtractorID == sharedCriticExtractorID)
        #expect(sharedActorTargetExtractorID == sharedCriticTargetExtractorID)
        #expect(sharedActorExtractorID != sharedActorTargetExtractorID)

        let actorDiff = maxParameterDiff(
            shared.actor.parameters(),
            shared.actorTarget.parameters()
        )
        let criticDiff = maxParameterDiff(
            shared.critic.parameters(),
            shared.criticTarget.parameters()
        )
        #expect(actorDiff < 1e-6)
        #expect(criticDiff < 1e-6)

        let unshared = try TD3Policy(
            observationSpace: observationSpace(),
            actionSpace: actionSpace(),
            featuresExtractor: .flatten,
            shareFeaturesExtractor: false
        )
        guard
            let unsharedCriticExtractor = unshared.critic.featuresExtractor,
            let unsharedCriticTargetExtractor = unshared.criticTarget.featuresExtractor
        else {
            Issue.record("TD3Policy critics should have feature extractors.")
            return
        }

        let unsharedActorExtractorID = ObjectIdentifier(unshared.actor.featuresExtractor as AnyObject)
        let unsharedCriticExtractorID = ObjectIdentifier(unsharedCriticExtractor as AnyObject)
        let unsharedActorTargetExtractorID = ObjectIdentifier(unshared.actorTarget.featuresExtractor as AnyObject)
        let unsharedCriticTargetExtractorID = ObjectIdentifier(unsharedCriticTargetExtractor as AnyObject)

        #expect(unsharedActorExtractorID != unsharedCriticExtractorID)
        #expect(unsharedActorTargetExtractorID != unsharedCriticTargetExtractorID)
    }

    @Test
    @MainActor
    func policyDelayDefersFirstActorUpdate() async throws {
        let env = try await Gymnazo.make("Pendulum")
        let td3 = try TD3(
            env: env,
            policyConfig: TD3PolicyConfig(
                featuresExtractor: .flatten,
                shareFeaturesExtractor: false
            ),
            algorithmConfig: TD3AlgorithmConfig(
                policyDelay: 2,
                actionNoise: .normal(std: 0.0)
            ),
            config: OffPolicyConfig(
                bufferSize: 64,
                learningStarts: 0,
                batchSize: 1,
                tau: 0.005,
                gamma: 0.99,
                trainFrequency: TrainFrequency(frequency: 1, unit: .step),
                gradientSteps: .fixed(1),
                targetUpdateInterval: 3,
                optimizeMemoryUsage: false,
                handleTimeoutTermination: true,
                useSDEAtWarmup: true,
                sdeSampleFreq: 5,
                sdeSupported: true
            ),
            seed: 7
        )

        let actorBefore = (await td3.actor).parameters()
        try await td3.learn(totalTimesteps: 1, callbacks: nil)
        let actorAfter = (await td3.actor).parameters()

        #expect(maxParameterDiff(actorBefore, actorAfter) < 1e-6)
        #expect(await td3.gradientSteps == 1)
    }

    @Test
    @MainActor
    func learnEvaluateAndActionBoundsSmoke() async throws {
        let trainingEnv = try await Gymnazo.make("Pendulum")
        let td3 = try TD3(
            env: trainingEnv,
            policyConfig: TD3PolicyConfig(
                featuresExtractor: .flatten,
                shareFeaturesExtractor: false
            ),
            algorithmConfig: TD3AlgorithmConfig(
                policyDelay: 2,
                targetPolicyNoise: 0.2,
                targetNoiseClip: 0.5,
                actionNoise: .ornsteinUhlenbeck(
                    std: 0.2,
                    theta: 0.15,
                    dt: 0.01,
                    initialNoise: 0.0
                )
            ),
            config: OffPolicyConfig(
                bufferSize: 256,
                learningStarts: 5,
                batchSize: 8,
                tau: 0.005,
                gamma: 0.99,
                trainFrequency: TrainFrequency(frequency: 1, unit: .step),
                gradientSteps: .fixed(1),
                targetUpdateInterval: 10,
                optimizeMemoryUsage: false,
                handleTimeoutTermination: true,
                useSDEAtWarmup: true,
                sdeSampleFreq: 4,
                sdeSupported: true
            ),
            seed: 123
        )

        try await td3.learn(totalTimesteps: 20, callbacks: nil)
        #expect(await td3.numTimesteps == 20)

        try await td3.evaluate(episodes: 1, deterministic: true, callbacks: nil)

        var envForBounds = try await Gymnazo.make("Pendulum")
        let obs = try envForBounds.reset(seed: 999).obs
        let action = await td3(observation: obs, deterministic: false)

        guard let box = envForBounds.actionSpace as? Box else {
            Issue.record("Pendulum action space should be Box.")
            return
        }

        let actionValues = action.reshaped([-1]).asArray(Float.self)
        let lowValues = box.low.reshaped([-1]).asArray(Float.self)
        let highValues = box.high.reshaped([-1]).asArray(Float.self)

        #expect(actionValues.count == lowValues.count)
        #expect(actionValues.count == highValues.count)
        for i in 0..<actionValues.count {
            #expect(actionValues[i] >= lowValues[i] - 1e-4)
            #expect(actionValues[i] <= highValues[i] + 1e-4)
        }
    }
}
