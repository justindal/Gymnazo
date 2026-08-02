import Testing

@testable import Gymnazo

private struct WrapperTestEnvironment: ValueEnvironment {
    let valueObservationSpace = try! DiscreteValueSpace(n: 4)
    let valueActionSpace = try! DiscreteValueSpace(n: 2)

    private var value = 0
    private var needsReset = true

    mutating func resetValue(
        seed _: UInt64?,
        options _: ValueResetOptions?
    ) throws -> ValueReset<Int> {
        value = 0
        needsReset = false
        return ValueReset(observation: value, info: ["reset": .bool(true)])
    }

    mutating func stepValue(_ action: Int) throws -> ValueStep<Int> {
        guard !needsReset else {
            throw GymnazoError.stepBeforeReset
        }
        guard valueActionSpace.containsValue(action) else {
            throw GymnazoError.invalidAction("Expected action zero or one.")
        }
        value += 1
        let terminated = value == 3
        needsReset = terminated
        return ValueStep(
            observation: value,
            reward: Double(action + 1),
            terminated: terminated,
            truncated: false,
            info: ["value": .discrete(value)]
        )
    }

    mutating func close() {
        needsReset = true
    }
}

@Suite("Typed environment wrappers")
struct ValueWrapperTests {
    @Test
    func timeLimitAddsTruncationWithoutErasingTermination() throws {
        var limited = try WrapperTestEnvironment().timeLimited(2)
        _ = try limited.resetValue(seed: 7)
        _ = try limited.stepValue(0)
        let truncated = try limited.stepValue(0)

        #expect(!truncated.terminated)
        #expect(truncated.truncated)
        #expect(truncated.info[EnvInfoKey.timeLimitTruncated] == .bool(true))

        var terminalLimit = try WrapperTestEnvironment().timeLimited(3)
        _ = try terminalLimit.resetValue(seed: 7)
        _ = try terminalLimit.stepValue(0)
        _ = try terminalLimit.stepValue(0)
        let terminal = try terminalLimit.stepValue(0)

        #expect(terminal.terminated)
        #expect(terminal.truncated)
        #expect(terminal.info[EnvInfoKey.timeLimitTruncated] == .bool(false))
    }

    @Test
    func transformsPreserveUnchangedPartsOfStep() throws {
        let outputSpace = try DiscreteValueSpace(n: 8)
        let inputSpace = try DiscreteValueSpace(n: 2)
        var environment = WrapperTestEnvironment()
            .actionsTransformed(from: inputSpace) { 1 - $0 }
            .rewardsTransformed { $0 * 2 }
            .observationsTransformed(to: outputSpace) { $0 * 2 }

        let reset = try environment.resetValue(seed: 9)
        let step = try environment.stepValue(0)

        #expect(reset.observation == 0)
        #expect(step.observation == 2)
        #expect(step.reward == 4)
        #expect(step.info["value"] == .discrete(1))
    }

    @Test
    func sameStepAutoResetPreservesTerminalPayload() throws {
        var environment = WrapperTestEnvironment().autoReset(mode: .sameStep)
        _ = try environment.resetValue(seed: 11)
        _ = try environment.stepValue(0)
        _ = try environment.stepValue(0)
        let terminal = try environment.stepValue(0)

        #expect(terminal.terminated)
        #expect(terminal.observation == 0)
        #expect(terminal.info[EnvInfoKey.finalObservation] == .discrete(3))
        #expect(terminal.info[EnvInfoKey.finalInfo] == .dictionary(["value": .discrete(3)]))

        let next = try environment.stepValue(0)
        #expect(next.observation == 1)
    }

    @Test
    func nextStepAutoResetDoesNotConsumeAction() throws {
        var environment = WrapperTestEnvironment().autoReset(mode: .nextStep)
        _ = try environment.resetValue(seed: 13)
        _ = try environment.stepValue(0)
        _ = try environment.stepValue(0)
        let terminal = try environment.stepValue(0)
        let reset = try environment.stepValue(1)
        let next = try environment.stepValue(1)

        #expect(terminal.observation == 3)
        #expect(reset.observation == 0)
        #expect(reset.reward == 0)
        #expect(next.observation == 1)
        #expect(next.reward == 2)
    }

    @Test
    func statisticsUseBoundedQueuesAndTypedInfo() throws {
        var environment = try WrapperTestEnvironment().recordingStatistics(bufferLength: 1)

        for _ in 0..<2 {
            _ = try environment.resetValue(seed: 17)
            _ = try environment.stepValue(0)
            _ = try environment.stepValue(0)
            let terminal = try environment.stepValue(0)

            guard case .dictionary(let statistics) = terminal.info[EnvInfoKey.episode] else {
                Issue.record("Expected typed episode statistics.")
                return
            }
            #expect(statistics["r"] == .double(3))
            #expect(statistics["l"] == .discrete(3))
        }

        #expect(environment.episodeCount == 2)
        #expect(environment.returnQueue == [3])
        #expect(environment.lengthQueue == [3])
        #expect(environment.timeQueue.count == 1)
    }
}
