import Testing
@testable import ExploreRLCore

@Suite("Gymnasium registration")
struct GymnasiumRegistrationTests {
    @Test
    @MainActor
    func testStartRegistersFrozenLake() async throws {
        Gymnasium.start()
        let contains = Gymnasium.registry.keys.contains("FrozenLake-v1")
        #expect(contains)
    }
    
    @Test
    @MainActor
    func testStartIsIdempotent() async throws {
        Gymnasium.start()
        let count1: Int = Gymnasium.registry.count
        Gymnasium.start()
        let count2: Int = Gymnasium.registry.count
        #expect(count1 == count2)
    }
}

