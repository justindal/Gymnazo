import Testing
import MLX

@testable import Gymnazo

@Suite("Gymnazo wrappers and env basics")
struct GymnazoTests {
    func makeFrozenLake(isSlippery: Bool, renderMode: RenderMode? = nil) async throws -> FrozenLake {
        var options: EnvOptions = ["is_slippery": isSlippery]
        if let renderMode {
            options["render_mode"] = renderMode.rawValue
        }
        let env = try await Gymnazo.make("FrozenLake", options: options)
        guard let frozenLake = env.unwrapped as? FrozenLake else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "FrozenLake",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return frozenLake
    }

    @Test
    func testTimeLimitAndEpisodeStats() async throws {
        let env = try await makeFrozenLake(isSlippery: false)
        let timeLimited = try TimeLimit(env: env, maxEpisodeSteps: 1)
        var recorder = try RecordEpisodeStatistics(
            env: timeLimited, bufferLength: 10, statsKey: "episode")
        _ = try recorder.reset(seed: 42)
        let result = try recorder.step(MLXArray(Int32(1)))
        #expect(result.truncated == true)
        #expect(result.terminated == false)
        #expect(result.info["episode"] != nil)
    }

    @Test
    @MainActor
    func testRenderPassThroughAnsi() async throws {
        var env = try await makeFrozenLake(isSlippery: false, renderMode: .ansi)
        _ = try env.reset(seed: 123)
        try env.render()
    }
}
