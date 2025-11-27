import Testing
@testable import Gymnazo

@Suite("Gymnazo registration")
struct GymnazoRegistrationTests {
    @Test
    @MainActor
    func testStartRegistersFrozenLake() async throws {
        Gymnazo.start()
        let contains = Gymnazo.registry.keys.contains("FrozenLake-v1")
        #expect(contains)
    }
    
    @Test
    @MainActor
    func testStartIsIdempotent() async throws {
        Gymnazo.start()
        let count1: Int = Gymnazo.registry.count
        Gymnazo.start()
        let count2: Int = Gymnazo.registry.count
        #expect(count1 == count2)
    }
}

