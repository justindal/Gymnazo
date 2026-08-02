import MLX
import Testing

@testable import Gymnazo

@Suite("Golden environment traces", .serialized)
struct GoldenEnvironmentTraceTests {
    @Test
    func frozenLakeTraceCoversStateRewardTerminationInfoAndRendering() throws {
        var environment = try FrozenLakeValueEnvironment(isSlippery: false)
        let reset = try environment.resetValue(seed: 101)
        #expect(reset.observation == 0)
        #expect(reset.info == ["prob": .double(1)])

        let actions = [2, 2, 1, 1, 1, 2]
        let expectedStates = [1, 2, 6, 10, 14, 15]
        let expectedRewards = [0.0, 0, 0, 0, 0, 1]

        for index in actions.indices {
            let step = try environment.stepValue(actions[index])
            #expect(step.observation == expectedStates[index])
            #expect(step.reward == expectedRewards[index])
            #expect(step.terminated == (index == actions.count - 1))
            #expect(step.truncated == false)
            #expect(step.info == ["prob": .double(1)])
        }

        var rendered = try FrozenLake(renderMode: .ansi, isSlippery: false)
        _ = try rendered.reset(seed: 101)
        for action in actions {
            _ = try rendered.step(MLXArray(Int32(action)))
        }
        guard case .ansi(let output) = try rendered.render() else {
            Issue.record("Expected an ANSI FrozenLake frame.")
            return
        }
        #expect(output == "S F F F\nF H F H\nF F F H\nH F F P")
    }

    @Test
    func cliffWalkingTraceCoversCliffPenaltyAndReset() throws {
        var environment = try CliffWalkingValueEnvironment(isSlippery: false)
        let reset = try environment.resetValue(seed: 202)
        #expect(reset.observation == 36)

        let up = try environment.stepValue(0)
        #expect(up.observation == 24)
        #expect(up.reward == -1)
        #expect(up.terminated == false)

        let right = try environment.stepValue(1)
        #expect(right.observation == 25)
        #expect(right.reward == -1)

        let cliff = try environment.stepValue(2)
        #expect(cliff.observation == 36)
        #expect(cliff.reward == -100)
        #expect(cliff.terminated == false)
        #expect(cliff.truncated == false)
        #expect(cliff.info == ["prob": .double(1)])
    }
}
