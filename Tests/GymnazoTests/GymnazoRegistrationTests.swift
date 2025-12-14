import Testing
@testable import Gymnazo

@Suite("Gymnazo registration")
struct GymnazoRegistrationTests {
    @Test
    @MainActor
    func testMakeRegistersEnvironments() async throws {
        // Calling make triggers lazy initialization
        let env = Gymnazo.make("FrozenLake", kwargs: ["is_slippery": false])
        #expect(env.spec?.id == "FrozenLake")
        
        // Registry should now contain registered environments
        let contains = Gymnazo.registry.keys.contains("FrozenLake")
        #expect(contains)
    }
}

