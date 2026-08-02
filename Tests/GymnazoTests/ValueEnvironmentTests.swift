import Testing

@testable import Gymnazo

private struct CounterValueSpace: ValueSpace {
    func containsValue(_ value: Int) -> Bool {
        (0...3).contains(value)
    }
}

private struct CounterValueEnvironment: ValueEnvironment {
    let valueObservationSpace = CounterValueSpace()
    let valueActionSpace = CounterValueSpace()
    private var value = 0

    mutating func resetValue(
        seed _: UInt64?,
        options _: ValueResetOptions?
    ) throws -> ValueReset<Int> {
        value = 0
        return ValueReset(observation: value)
    }

    mutating func stepValue(_ action: Int) throws -> ValueStep<Int> {
        value = action
        return ValueStep(
            observation: value,
            reward: Double(value),
            terminated: value == 3,
            truncated: false,
            info: ["state": .discrete(value)]
        )
    }

    mutating func close() {}
}

@Suite("Value environment foundation")
struct ValueEnvironmentTests {
    @Test
    func typedStateAndActionRemainSendableValues() throws {
        var environment = CounterValueEnvironment()
        let reset = try environment.resetValue(seed: 42)
        let step = try environment.stepValue(3)

        #expect(reset.observation == 0)
        #expect(step.terminated)
        #expect(step.info["state"] == .discrete(3))
    }

    @Test
    func checkerEnforcesEpisodeStateAndSpaceMembership() throws {
        var environment = CheckedValueEnvironment(CounterValueEnvironment())

        #expect(throws: GymnazoError.self) {
            try environment.stepValue(0)
        }

        _ = try environment.resetValue(seed: 42)

        #expect(throws: GymnazoError.self) {
            try environment.stepValue(4)
        }

        let step = try environment.stepValue(3)
        #expect(step.terminated)

        #expect(throws: GymnazoError.self) {
            try environment.stepValue(0)
        }

        environment.close()
        environment.close()

        #expect(throws: GymnazoError.self) {
            try environment.resetValue(seed: nil)
        }
    }
}
