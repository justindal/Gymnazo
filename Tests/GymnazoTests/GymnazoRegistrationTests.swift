import Testing
@testable import Gymnazo

@Suite("Gymnazo registration")
struct GymnazoRegistrationTests {
    @Test
    @MainActor
    func testMakeRegistersEnvironments() async throws {
        // Calling make triggers lazy initialization
        let env = Gymnazo.make("FrozenLake-v1", kwargs: ["is_slippery": false])
        #expect(env.spec?.id == "FrozenLake-v1")
        
        // Registry should now contain registered environments
        let contains = Gymnazo.registry.keys.contains("FrozenLake-v1")
        #expect(contains)
    }
}

