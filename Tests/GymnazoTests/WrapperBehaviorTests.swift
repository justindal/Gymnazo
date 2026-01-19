import Testing

@testable import Gymnazo

@Suite("Wrapper behaviors")
struct WrapperBehaviorTests {
    func makeFrozenLake(isSlippery: Bool) async throws -> FrozenLake {
        let env: AnyEnv<Int, Int> = try await Gymnazo.make(
            "FrozenLake",
            options: ["is_slippery": isSlippery]
        )
        guard let frozenLake = env.unwrapped as? FrozenLake else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "FrozenLake",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return frozenLake
    }

    @Test
    func testRecordEpisodeStatisticsAddsInfoOnTruncate() async throws {
        let env = try await makeFrozenLake(isSlippery: false)
        let timeLimited = try TimeLimit(env: env, maxEpisodeSteps: 1)
        var recorder = try RecordEpisodeStatistics(
            env: timeLimited, bufferLength: 4, statsKey: "episode")
        _ = try recorder.reset(seed: 99)
        let step = try recorder.step(1)
        #expect(step.truncated)
        #expect(step.info["episode"] != nil)
    }

    @Test
    @MainActor
    func testRenderAllowedBeforeResetWhenDisabled() async throws {
        var env: AnyEnv<Int, Int> = try await Gymnazo.make(
            "FrozenLake",
            disableRenderOrderEnforcing: true,
            options: ["render_mode": "ansi", "is_slippery": false]
        )
        try env.render()
        _ = try env.reset(seed: 1)
        try env.render()
    }
}
