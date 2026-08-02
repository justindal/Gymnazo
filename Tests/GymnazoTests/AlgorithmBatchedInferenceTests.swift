import Testing

@testable import Gymnazo

@Suite("Batched algorithm inference", .serialized)
struct AlgorithmBatchedInferenceTests {
    @Test
    @MainActor
    func dqnBatchesAtOnePolicyBoundary() async throws {
        let model = try DQN(
            observationSpace: Box(low: -1, high: 1, shape: [4]),
            actionSpace: Discrete(n: 2),
            policyConfig: DQNPolicyConfig(netArch: [16], featuresExtractor: .flatten),
            seed: 71
        )
        let observations = try [
            TensorValue([Float](repeating: 0, count: 4), shape: [4]),
            TensorValue([Float](repeating: 0.5, count: 4), shape: [4]),
        ]
        let batched = try await model.act(observations: observations)
        var scalar: [PolicyAction] = []
        for observation in observations {
            scalar.append(await model.act(observation: observation))
        }

        #expect(batched == scalar)
    }

    @Test
    @MainActor
    func ppoBatchesDiscreteActionsWithoutCollapsingTheBatch() async throws {
        let model = try PPO(
            observationSpace: Box(low: -1, high: 1, shape: [4]),
            actionSpace: Discrete(n: 2),
            policyConfig: PPOPolicyConfig(
                netArch: .shared([16]),
                featuresExtractor: .flatten
            ),
            seed: 73
        )
        let observations = try [
            TensorValue([Float](repeating: -0.5, count: 4), shape: [4]),
            TensorValue([Float](repeating: 0.5, count: 4), shape: [4]),
        ]
        let actions = try await model.act(observations: observations)

        #expect(actions.count == observations.count)
        #expect(
            actions.allSatisfy {
                guard case .discrete(let value) = $0 else { return false }
                return (0..<2).contains(value)
            })
    }

    @Test
    @MainActor
    func continuousAlgorithmsPreserveBatchAndActionDimensions() async throws {
        let sac = try SAC(
            observationSpace: Box(low: -1, high: 1, shape: [3]),
            actionSpace: Box(low: -2, high: 2, shape: [2]),
            networksConfig: SACNetworksConfig(
                actor: SACActorConfig(netArch: .shared([16]), featuresExtractor: .flatten),
                critic: SACCriticConfig(netArch: [16], featuresExtractor: .flatten)
            ),
            seed: 79
        )
        let td3 = try TD3(
            observationSpace: Box(low: -1, high: 1, shape: [3]),
            actionSpace: Box(low: -2, high: 2, shape: [2]),
            policyConfig: TD3PolicyConfig(
                netArch: .shared([16]),
                featuresExtractor: .flatten
            ),
            seed: 83
        )
        let observations = try [
            TensorValue([Float](repeating: -0.25, count: 3), shape: [3]),
            TensorValue([Float](repeating: 0.25, count: 3), shape: [3]),
        ]
        let sacActions = try await sac.act(observations: observations)
        let td3Actions = try await td3.act(observations: observations)

        for actions in [sacActions, td3Actions] {
            #expect(actions.count == observations.count)
            #expect(
                actions.allSatisfy {
                    guard case .continuous(let values) = $0 else { return false }
                    return values.count == 2 && values.allSatisfy { (-2...2).contains($0) }
                })
        }
    }

    @Test
    @MainActor
    func dqnCollectsAndInsertsTypedVectorTransitions() async throws {
        let model = try DQN(
            observationSpace: Box(low: -100, high: 100, shape: [4]),
            actionSpace: Discrete(n: 2),
            policyConfig: DQNPolicyConfig(netArch: [16], featuresExtractor: .flatten),
            config: DQNConfig(explorationInitialEps: 0, explorationFinalEps: 0),
            seed: 89
        )
        let session = try ValueCollectionSession(
            environments: [
                try CartPoleValueEnvironment(),
                try CartPoleValueEnvironment(),
            ]
        )
        let collection = try await model.collect(
            from: session,
            steps: 3,
            seed: 103,
            deterministic: true
        )
        var replay = try ReplayBuffer(
            observationSpace: Box(low: -100, high: 100, shape: [4]),
            actionSpace: Discrete(n: 2),
            config: .init(bufferSize: 10, seed: 107),
            numEnvs: 2
        )

        try replay.add(collection)

        #expect(collection.frames.count == 3)
        #expect(replay.count == 6)
    }
}
