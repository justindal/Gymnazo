import MLX
import Testing

@testable import Gymnazo

private struct ImageTrainingEnvironment: Env {
    let observationSpace: any Space = Box(low: 0, high: 1, shape: [36, 36, 1])
    let actionSpace: any Space
    var spec: EnvSpec?
    var renderMode: RenderMode?

    init(continuous: Bool = false) {
        actionSpace = continuous ? Box(low: -1, high: 1, shape: [1]) : Discrete(n: 2)
    }

    mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        Reset(obs: MLX.ones([36, 36, 1]) * 0.5)
    }

    mutating func step(_ action: MLXArray) throws -> Step {
        Step(obs: MLX.ones([36, 36, 1]) * 0.25, reward: 1, terminated: true, truncated: false)
    }
}

@Suite("CNN training", .serialized)
struct CNNTrainingTests {
    @Test
    @MainActor
    func dqnAndPPOCompileCNNUpdatesAcrossMultipleSteps() async throws {
        let dqn = try DQN(
            env: ImageTrainingEnvironment(),
            policyConfig: DQNPolicyConfig(
                netArch: [16], featuresExtractor: .natureCNN(featuresDim: 16)),
            config: DQNConfig(bufferSize: 8, learningStarts: 0, batchSize: 2), seed: 43
        )
        let ppo = try PPO(
            env: ImageTrainingEnvironment(),
            policyConfig: PPOPolicyConfig(
                netArch: .shared([16]), featuresExtractor: .natureCNN(featuresDim: 16)),
            config: PPOConfig(nSteps: 4, batchSize: 2, nEpochs: 2), seed: 43
        )
        let dqnBefore = await dqn.policyParameterValues()
        let ppoBefore = await ppo.policyParameterValues()
        try await dqn.learn(totalTimesteps: 6, callbacks: nil)
        try await ppo.learn(totalTimesteps: 8, callbacks: nil)
        let dqnAfter = await dqn.policyParameterValues()
        let ppoAfter = await ppo.policyParameterValues()
        #expect(dqnAfter != dqnBefore)
        #expect(ppoAfter != ppoBefore)
        #expect(dqnAfter.allSatisfy { $0.isFinite })
        #expect(ppoAfter.allSatisfy { $0.isFinite })
    }
    @Test
    @MainActor
    func sacAndTD3CompileCNNUpdatesAcrossMultipleSteps() async throws {
        let config = OffPolicyConfig(bufferSize: 8, learningStarts: 0, batchSize: 2)
        let sac = try SAC(
            env: ImageTrainingEnvironment(continuous: true),
            networksConfig: SACNetworksConfig(
                actor: SACActorConfig(
                    netArch: .shared([16]), featuresExtractor: .natureCNN(featuresDim: 16)),
                critic: SACCriticConfig(
                    netArch: [16], featuresExtractor: .natureCNN(featuresDim: 16))
            ), config: config, seed: 43
        )
        let td3 = try TD3(
            env: ImageTrainingEnvironment(continuous: true),
            policyConfig: TD3PolicyConfig(
                netArch: .shared([16]), featuresExtractor: .natureCNN(featuresDim: 16)),
            config: config, seed: 43
        )
        let sacBefore = await sac.criticTargetParameterValues()
        let td3Before = await td3.actorParameterValues()
        try await sac.learn(totalTimesteps: 6, callbacks: nil)
        try await td3.learn(totalTimesteps: 6, callbacks: nil)
        let sacAfter = await sac.criticTargetParameterValues()
        let td3After = await td3.actorParameterValues()
        #expect(sacAfter != sacBefore)
        #expect(td3After != td3Before)
        #expect(sacAfter.allSatisfy { $0.isFinite })
        #expect(td3After.allSatisfy { $0.isFinite })
    }

}
