import MLX
import Testing

@testable import Gymnazo

@Suite("Reward wrappers")
struct RewardWrappersTests {
    @Test
    func testTransformRewardApplies() async throws {
        let baseEnv = try await Gymnazo.make("CartPole")
        var env = baseEnv.rewardsTransformed { $0 * 2 }
        _ = try env.reset(seed: 0, options: nil)
        let result = try env.step(MLXArray(Int32(0)))
        #expect(result.reward == 2)
    }

    @Test
    func testNormalizeRewardIsFinite() async throws {
        let baseEnv = try await Gymnazo.make("CartPole")
        var env = baseEnv.rewardsNormalized()
        _ = try env.reset(seed: 0, options: nil)
        let result = try env.step(MLXArray(Int32(0)))
        #expect(result.reward.isFinite)
    }
}
