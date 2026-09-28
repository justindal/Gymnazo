import MLX
import Testing

@testable import Gymnazo

@Suite("Batched training", .serialized)
struct ValueTrainingTests {
    @Test
    @MainActor
    func dqnTrainsAcrossVectorFramesAndCountsTransitions() async throws {
        let model = try DQN(
            observationSpace: Box(low: -100, high: 100, shape: [4]), actionSpace: Discrete(n: 2),
            policyConfig: DQNPolicyConfig(netArch: [16], featuresExtractor: .flatten),
            config: DQNConfig(bufferSize: 32, learningStarts: 0, batchSize: 2), seed: 43
        )
        let before = await model.policyParameterValues()
        try await model.learn(
            environments: [try CartPoleValueEnvironment(), try CartPoleValueEnvironment()],
            totalTimesteps: 11
        )
        #expect(await model.timesteps == 12)
        #expect(await model.policyParameterValues() != before)
    }

    @Test
    @MainActor
    func ppoVectorLearningIsSeededAndUpdatesPolicy() async throws {
        func model() throws -> PPO {
            try PPO(
                observationSpace: Box(low: -100, high: 100, shape: [4]),
                actionSpace: Discrete(n: 2),
                policyConfig: PPOPolicyConfig(netArch: .shared([16]), featuresExtractor: .flatten),
                config: PPOConfig(nSteps: 4, batchSize: 3, nEpochs: 2), seed: 43
            )
        }
        let first = try model()
        let second = try model()
        let before = await first.policyParameterValues()
        try await first.learn(
            environments: [try CartPoleValueEnvironment(), try CartPoleValueEnvironment()],
            totalTimesteps: 16
        )
        try await second.learn(
            environments: [try CartPoleValueEnvironment(), try CartPoleValueEnvironment()],
            totalTimesteps: 16
        )
        #expect(await first.timesteps == 16)
        #expect(await first.updates == 4)
        #expect(await first.policyParameterValues() != before)
        let firstWeights = await first.policyParameterValues()
        let secondWeights = await second.policyParameterValues()
        let maximumDifference = zip(firstWeights, secondWeights).map { abs($0 - $1) }.max() ?? 0
        #expect(
            firstWeights == secondWeights, "Maximum seeded weight difference: \(maximumDifference)")
    }

    @Test
    @MainActor
    func continuousVectorLearningUpdatesSACAndTD3() async throws {
        let config = OffPolicyConfig(bufferSize: 32, learningStarts: 2, batchSize: 2)
        let sac = try SAC(
            observationSpace: Box(low: -10, high: 10, shape: [3]),
            actionSpace: Box(low: -2, high: 2, shape: [1]),
            networksConfig: SACNetworksConfig(
                actor: SACActorConfig(netArch: .shared([16]), featuresExtractor: .flatten),
                critic: SACCriticConfig(netArch: [16], featuresExtractor: .flatten)
            ), config: config, seed: 43
        )
        let td3 = try TD3(
            observationSpace: Box(low: -10, high: 10, shape: [3]),
            actionSpace: Box(low: -2, high: 2, shape: [1]),
            policyConfig: TD3PolicyConfig(netArch: .shared([16]), featuresExtractor: .flatten),
            config: config, seed: 43
        )
        let sacBefore = await sac.criticTargetParameterValues()
        let td3Before = await td3.actorParameterValues()
        try await sac.learn(
            environments: [try PendulumValueEnvironment(), try PendulumValueEnvironment()],
            totalTimesteps: 12
        )
        try await td3.learn(
            environments: [try PendulumValueEnvironment(), try PendulumValueEnvironment()],
            totalTimesteps: 12
        )
        #expect(await sac.criticTargetParameterValues() != sacBefore)
        #expect(await td3.actorParameterValues() != td3Before)
        #expect(await sac.timesteps == 12)
        #expect(await td3.timesteps == 12)
    }
    @Test
    @MainActor
    func vectorTrainingRejectsObservationMismatchBeforeMLXExecution() async throws {
        let model = try DQN(
            observationSpace: Box(low: -100, high: 100, shape: [3]), actionSpace: Discrete(n: 2),
            policyConfig: DQNPolicyConfig(netArch: [16], featuresExtractor: .flatten), seed: 43
        )
        await #expect(throws: GymnazoError.self) {
            try await model.learn(environments: [try CartPoleValueEnvironment()], totalTimesteps: 2)
        }
        #expect(await model.timesteps == 0)
    }

    @Test
    func ornsteinUhlenbeckNoiseResetsOnlyFinishedSlots() {
        let noise = TD3OrnsteinUhlenbeckActionNoise(std: 0, theta: 1, dt: 0.5, initialNoise: 1)
        _ = noise.sample(shape: [2, 1], key: MLX.key(1))
        noise.reset(indices: [true, false])
        let result = noise.sample(shape: [2, 1], key: MLX.key(2)).asArray(Float.self)
        #expect(result == [0.5, 0.25])
    }

    @Test
    @MainActor
    func vectorTrainingHonorsProgressCancellationBeforeUpdating() async throws {
        let model = try DQN(
            observationSpace: Box(low: -100, high: 100, shape: [4]), actionSpace: Discrete(n: 2),
            policyConfig: DQNPolicyConfig(netArch: [16], featuresExtractor: .flatten),
            config: DQNConfig(bufferSize: 32, learningStarts: 0, batchSize: 2), seed: 43
        )
        let before = await model.policyParameterValues()
        try await model.learn(
            environments: [try CartPoleValueEnvironment(), try CartPoleValueEnvironment()],
            totalTimesteps: 20, callbacks: LearnCallbacks(onStep: { _, _, _ in false })
        )
        #expect(await model.timesteps == 2)
        #expect(await model.policyParameterValues() == before)
    }

}
