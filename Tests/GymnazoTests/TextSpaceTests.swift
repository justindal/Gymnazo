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
            let s = space.sample(key: key)
            #expect(space.contains(s))
            #expect(s.count <= 12)
        }
    }

    @Test
    func testContainsRejectsInvalidCharacters() async throws {
        let space = TextSpace(minLength: 1, maxLength: 5, charset: Array("ab".map { $0 }))
        #expect(space.contains("ab"))
        #expect(space.contains("aba"))
        #expect(space.contains("abc") == false)
    }

    @Test
    func testContainsRejectsInvalidLength() async throws {
        let space = TextSpace(minLength: 2, maxLength: 3, charset: Array("ab".map { $0 }))
        #expect(space.contains("a") == false)
        #expect(space.contains("ab"))
        #expect(space.contains("aba"))
        #expect(space.contains("abaa") == false)
    }
}

