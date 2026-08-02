import Testing

@testable import Gymnazo

struct EpisodeSpace: ValueSpace {
    func containsValue(_ value: Int) -> Bool {
        (0...2).contains(value)
    }
}

struct EpisodeEnvironment: ValueEnvironment {
    let valueObservationSpace = EpisodeSpace()
    let valueActionSpace = EpisodeSpace()
    private var state = 0

    mutating func resetValue(
        seed _: UInt64?,
        options _: ValueResetOptions?
    ) throws -> ValueReset<Int> {
        state = 0
        return ValueReset(observation: state, info: ["reset": .bool(true)])
    }

    mutating func stepValue(_ action: Int) throws -> ValueStep<Int> {
        state = action
        return ValueStep(
            observation: state,
            reward: Double(action),
            terminated: action == 2,
            truncated: false,
            info: ["step": .discrete(action)]
        )
    }

    mutating func close() {}
}

@Suite("Synchronous value vector environment")
struct SyncValueVectorEnvironmentTests {
    @Test
    func steppingBeforeResetFails() throws {
        var environment = try SyncValueVectorEnvironment(environments: [EpisodeEnvironment()])

        #expect(throws: GymnazoError.self) {
            try environment.step([1])
        }
    }

    @Test
    func sameStepAutoresetKeepsTerminalDataTyped() throws {
        var environment = try SyncValueVectorEnvironment(
            environments: [EpisodeEnvironment(), EpisodeEnvironment()],
            autoresetMode: .sameStep
        )

        _ = try environment.reset(seed: 12)
        let result = try environment.step([2, 1])

        #expect(result.observations == [0, 1])
        #expect(result.rewards == [2, 1])
        #expect(result.terminations == [true, false])
        #expect(result.finalObservations == [2, nil])
        #expect(result.finalInfos[0]?["step"] == .discrete(2))
        #expect(result.resetInfos[0]?["reset"] == .bool(true))
    }

    @Test
    func nextStepAutoresetDoesNotConsumeTheNextAction() throws {
        var environment = try SyncValueVectorEnvironment(
            environments: [EpisodeEnvironment()],
            autoresetMode: .nextStep
        )

        _ = try environment.reset()
        _ = try environment.step([2])
        let resetStep = try environment.step([1])

        #expect(resetStep.observations == [0])
        #expect(resetStep.rewards == [0])
        #expect(resetStep.terminations == [false])
        #expect(resetStep.infos[0]["reset"] == .bool(true))
    }

    @Test
    func disabledAutoresetRequiresResetAfterTermination() throws {
        var environment = try SyncValueVectorEnvironment(
            environments: [EpisodeEnvironment()],
            autoresetMode: .disabled
        )

        _ = try environment.reset()
        _ = try environment.step([2])

        #expect(throws: GymnazoError.self) {
            try environment.step([1])
        }
    }
}
