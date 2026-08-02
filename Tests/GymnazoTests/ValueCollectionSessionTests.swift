import Testing

@testable import Gymnazo

@Suite("Typed collection session")
struct ValueCollectionSessionTests {
    @Test
    func collectsOrderedBatchesWithSameStepTerminalData() async throws {
        let session = try ValueCollectionSession(
            environments: [EpisodeEnvironment(), EpisodeEnvironment()]
        )

        let collection = try await session.collect(steps: 3, seed: 41) { observations in
            observations.map { min($0 + 1, 2) }
        }

        #expect(collection.reset.observations == [0, 0])
        #expect(collection.frames.count == 3)
        #expect(collection.frames.allSatisfy { $0.actions.count == 2 })

        let terminal = collection.frames[1].step
        #expect(terminal.terminations == [true, true])
        #expect(terminal.finalObservations == [2, 2])
        #expect(terminal.observations == [0, 0])
        #expect(terminal.resetInfos.allSatisfy { $0 != nil })
    }

    @Test
    func rejectsInvalidStepCounts() async throws {
        let session = try ValueCollectionSession(
            environments: [EpisodeEnvironment()]
        )

        await #expect(throws: GymnazoError.self) {
            try await session.collect(steps: -1) { _ in [0] }
        }
    }
}
