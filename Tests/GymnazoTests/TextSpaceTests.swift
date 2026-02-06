import Testing
import MLX
@testable import Gymnazo

@Suite("Text space sampling and membership")
struct TextSpaceTests {
    @Test
    func testSampleContains() async throws {
        let space = TextSpace(minLength: 0, maxLength: 12)
        var key = MLX.key(42)
        for _ in 0..<20 {
            let (k, _) = MLX.split(key: key)
            key = k
            let flat = space.sample(key: key)
            #expect(space.contains(flat))
            let s = space.sampleString(key: key)
            #expect(s.count <= 12)
        }
    }

    @Test
    func testContainsRejectsInvalidCharacters() async throws {
        let space = TextSpace(minLength: 1, maxLength: 5, charset: Array("ab".map { $0 }))
        #expect(space.containsString("ab"))
        #expect(space.containsString("aba"))
        #expect(space.containsString("abc") == false)
    }

    @Test
    func testContainsRejectsInvalidLength() async throws {
        let space = TextSpace(minLength: 2, maxLength: 3, charset: Array("ab".map { $0 }))
        #expect(space.containsString("a") == false)
        #expect(space.containsString("ab"))
        #expect(space.containsString("aba"))
        #expect(space.containsString("abaa") == false)
    }
}
