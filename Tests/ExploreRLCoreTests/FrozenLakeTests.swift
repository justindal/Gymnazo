import Testing
@testable import ExploreRLCore

@Suite("FrozenLake environment")
struct FrozenLakeTests {
    @Test
    func testResetDeterminismWithSeed() async throws {
        let env = FrozenLake(isSlippery: false)
        let r1 = env.reset(seed: 7)
        let r2 = env.reset(seed: 7)
        #expect(r1.obs == r2.obs)
        #expect((r1.info["prob"] as? Double) == 1.0)
    }
    
    @Test
    func testStepRightNonSlippery() async throws {
        let env = FrozenLake(isSlippery: false)
        _ = env.reset(seed: 0)
        let result = env.step(2) // right
        #expect(result.obs == 1) // from (0,0) -> (0,1)
        #expect(result.terminated == false)
        #expect(result.truncated == false)
    }
    
    @Test
    @MainActor
    func testGymnasiumMakeFrozenLake() async throws {
        Gymnasium.start()
        let env = Gymnasium.make("FrozenLake-v1", kwargs: ["is_slippery": false])
        let fl = env.unwrapped as! FrozenLake
        _ = fl.reset(seed: 123)
        let s = fl.step(2)
        #expect(s.obs == 1)
    }
}

