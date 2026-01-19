import Testing
@testable import Gymnazo

@Suite("Gymnazo registration")
struct GymnazoRegistrationTests {
    @Test
    @MainActor
    func testMakeRegistersEnvironments() async throws {
        let env: AnyEnv<Int, Int> = try await Gymnazo.make(
            "FrozenLake",
            options: ["is_slippery": false]
        )
        #expect(env.spec?.id == "FrozenLake")
        
        let specs = await Gymnazo.registry()
        let contains = specs.keys.contains("FrozenLake")
        #expect(contains)
    }
}

