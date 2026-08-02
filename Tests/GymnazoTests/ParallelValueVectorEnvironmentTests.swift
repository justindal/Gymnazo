import Testing

@testable import Gymnazo

@Suite("Parallel value vector environment")
struct ParallelValueVectorEnvironmentTests {
    @Test
    func parallelCollectionPreservesEnvironmentOrder() async throws {
        let environment = try ParallelValueVectorEnvironment<EpisodeEnvironment>(
            factories: [{ EpisodeEnvironment() }, { EpisodeEnvironment() }],
            autoresetMode: .sameStep
        )

        _ = try await environment.reset(seed: 24)
        let result = try await environment.step([2, 1])

        #expect(result.observations == [0, 1])
        #expect(result.rewards == [2, 1])
        #expect(result.finalObservations == [2, nil])
    }

    @Test
    func parallelCollectionRequiresResetBeforeStep() async throws {
        let environment = try ParallelValueVectorEnvironment<EpisodeEnvironment>(
            factories: [{ EpisodeEnvironment() }]
        )

        await #expect(throws: GymnazoError.self) {
            try await environment.step([1])
        }
    }
}
