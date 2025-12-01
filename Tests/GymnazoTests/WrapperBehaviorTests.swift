import Testing
@testable import Gymnazo

@Suite("Wrapper behaviors")
struct WrapperBehaviorTests {
    @Test
    func testRecordEpisodeStatisticsAddsInfoOnTruncate() async throws {
        let env = FrozenLake(isSlippery: false)
        let timeLimited = TimeLimit(env: env, maxEpisodeSteps: 1)
        var recorder = RecordEpisodeStatistics(env: timeLimited, bufferLength: 4, statsKey: "episode")
        _ = recorder.reset(seed: 99)
        let step = recorder.step(1)
        #expect(step.truncated)
        #expect(step.info["episode"] != nil)
    }
    
    @Test
    @MainActor
    func testRenderAllowedBeforeResetWhenDisabled() async throws {
        var env = Gymnazo.make(
            "FrozenLake-v1",
            disableRenderOrderEnforcing: true,
            kwargs: ["render_mode": "ansi", "is_slippery": false]
        )
        // should not crash due to disabled order enforcement
        env.render()
        _ = env.reset(seed: 1)
        env.render()
    }
}

