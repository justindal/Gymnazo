import Testing
import MLX

@testable import Gymnazo

@Suite("Wrapper behaviors")
struct WrapperBehaviorTests {
    func makeFrozenLake(isSlippery: Bool) async throws -> any Env {
        try await Gymnazo.make(
            "FrozenLake",
            options: ["is_slippery": isSlippery]
        )
    }

    @Test
    func testRecordEpisodeStatisticsAddsInfoOnTruncate() async throws {
        let base = try await makeFrozenLake(isSlippery: false)
        let timeLimited = try TimeLimit(env: base, maxEpisodeSteps: 1)
        var recorder = try RecordEpisodeStatistics(
            env: timeLimited, bufferLength: 4, statsKey: "episode")
        _ = try recorder.reset(seed: 99)
        let step = try recorder.step(MLXArray(Int32(1)))
        #expect(step.truncated)
        #expect(step.info["episode"] != nil)
    }

    @Test
    @MainActor
    func testRenderAllowedBeforeResetWhenDisabled() async throws {
        var env = try await Gymnazo.make(
            "FrozenLake",
            disableRenderOrderEnforcing: true,
            options: ["render_mode": "ansi", "is_slippery": false]
        )
        try env.render()
        _ = try env.reset(seed: 1)
        try env.render()
    }
}
