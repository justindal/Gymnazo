import Testing
import MLX
@testable import Gymnazo

@Suite("Reward wrappers")
struct RewardWrappersTests {
    @Test
    func testTransformRewardApplies() async throws {
        var baseEnv: AnyEnv<MLXArray, Int> = try await Gymnazo.make("CartPole")
        var env = try baseEnv.rewardsTransformed { $0 * 2 }
        _ = try env.reset(seed: 0, options: nil)
        let result = try env.step(0)
        #expect(result.reward == 2)
    }

    @Test
    func testNormalizeRewardIsFinite() async throws {
        var baseEnv: AnyEnv<MLXArray, Int> = try await Gymnazo.make("CartPole")
        var env = try baseEnv.rewardsNormalized()
        _ = try env.reset(seed: 0, options: nil)
        let result = try env.step(0)
        #expect(result.reward.isFinite)
    }
}

