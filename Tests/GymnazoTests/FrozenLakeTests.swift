import Testing
@testable import Gymnazo

@Suite("FrozenLake environment")
struct FrozenLakeTests {
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
    func testResetDeterminismWithSeed() async throws {
        let env = try await makeFrozenLake(isSlippery: false)
        let r1 = try env.reset(seed: 7)
        let r2 = try env.reset(seed: 7)
        #expect(r1.obs == r2.obs)
        #expect(r1.info["prob"]?.double == 1.0)
    }
    
    @Test
    func testStepRightNonSlippery() async throws {
        let env = try await makeFrozenLake(isSlippery: false)
        _ = try env.reset(seed: 0)
        let result = try env.step(2)
        #expect(result.obs == 1)
        #expect(result.terminated == false)
        #expect(result.truncated == false)
    }
    
    @Test
    @MainActor
    func testGymnazoMakeFrozenLake() async throws {
        let env: AnyEnv<Int, Int> = try await Gymnazo.make(
            "FrozenLake",
            options: ["is_slippery": false]
        )
        let fl = env.unwrapped as! FrozenLake
        _ = try fl.reset(seed: 123)
        let s = try fl.step(2)
        #expect(s.obs == 1)
    }
}

