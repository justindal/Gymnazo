import MLX
import Testing

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
        #expect(!space.contains(MLXArray([Float(0), Float(1), Float(0), Float(1)])))
    }

    @Test
    func maskForcesBitsAndProbabilityControlsEachElement() {
        let space = MultiBinary(n: 4)
        let mask = MLXArray([Int32(0), Int32(1), Int32(2), Int32(0)])
        let masked = space.sample(key: MLX.key(7), mask: mask, probability: nil)
        eval(masked)
        let maskedValues = masked.asArray(Int32.self)
        #expect(maskedValues[0] == 0)
        #expect(maskedValues[1] == 1)
        #expect(maskedValues[3] == 0)

        let probabilities = MLXArray([Float(0), Float(1), Float(0), Float(1)])
        let weighted = space.sample(
            key: MLX.key(8),
            mask: nil,
            probability: probabilities
        )
        eval(weighted)
        #expect(weighted.asArray(Int32.self) == [0, 1, 0, 1])
    }
}
