import Testing
import MLX
@testable import Gymnazo

@Suite("MultiBinary space sampling and membership")
struct MultiBinarySpaceTests {
    @Test
    func testSampleContains() async throws {
        let space = MultiBinary(shape: [2, 3])
        var key = MLX.key(123)
        for _ in 0..<10 {
            let (k, _) = MLX.split(key: key)
            key = k
            let x = space.sample(key: key)
            #expect(space.contains(x))
            #expect(x.shape == [2, 3])
        }
    }

    @Test
    func testContainsRejectsOutOfRange() async throws {
        let space = MultiBinary(n: 4)
        let ok = MLXArray([Int32](arrayLiteral: 0, 1, 1, 0))
        let bad = MLXArray([Int32](arrayLiteral: 0, 2, 1, 0))
        #expect(space.contains(ok))
        #expect(space.contains(bad) == false)
    }
}

