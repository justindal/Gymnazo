import Testing
@testable import Gymnazo

@Suite("Gymnazo wrappers and env basics")
struct GymnazoTests {
    @Test
    func testTimeLimitAndEpisodeStats() async throws {
        // deterministic for stability
        let env = FrozenLake(isSlippery: false)
        let timeLimited = TimeLimit(env: env, maxEpisodeSteps: 1)
        var recorder = RecordEpisodeStatistics(env: timeLimited, bufferLength: 10, statsKey: "episode")
        _ = recorder.reset(seed: 42)
        let result = recorder.step(1) // any action
        #expect(result.truncated == true)
        #expect(result.terminated == false)
        // stats should be appended on truncation
        #expect(result.info["episode"] != nil)
    }

    @Test
    @MainActor
    func testRenderPassThroughAnsi() async throws {
        var env = Gymnazo.make(
            "FrozenLake",
            kwargs: ["render_mode": "ansi", "is_slippery": false]
        )
        _ = env.reset(seed: 123)
        // should not crash; we don't assert output here.
        env.render()
    }
}
