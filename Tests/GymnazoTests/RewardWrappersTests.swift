import Testing
import MLX
@testable import Gymnazo

@Suite("Reward wrappers")
struct RewardWrappersTests {
    @Test
    func testTransformRewardApplies() async throws {
        var env = CartPole().rewardsTransformed { $0 * 2 }
        _ = env.reset(seed: 0, options: nil)
        let result = env.step(0)
        #expect(result.reward == 2)
    }

    @Test
    func testNormalizeRewardIsFinite() async throws {
        var env = CartPole().rewardsNormalized()
        _ = env.reset(seed: 0, options: nil)
        let result = env.step(0)
        #expect(result.reward.isFinite)
    }
}

