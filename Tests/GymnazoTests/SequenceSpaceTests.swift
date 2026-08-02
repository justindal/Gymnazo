import MLX
import Testing

@testable import Gymnazo

@Suite("Sequence space sampling and membership")
struct SequenceSpaceTests {
    @Test
    func testSampleContains() async throws {
        let inner = Box(low: -1.0, high: 1.0, shape: [2], dtype: .float32)
        let space = SequenceSpace(space: inner, minLength: 0, maxLength: 5)
        var key = MLX.key(777)
        for _ in 0..<10 {
            let (k, _) = MLX.split(key: key)
            key = k
            let flat = space.sample(key: key)
            #expect(space.contains(flat))
            let sample = space.sampleSequence(key: key, mask: nil, probability: nil)
            #expect(sample.values.shape == [5, 2])
            #expect(sample.mask.shape == [5])
        }
    }

    @Test
    func testContainsRejectsNonPrefixMask() async throws {
        let inner = MultiBinary(n: 3)
        let space = SequenceSpace(space: inner, minLength: 0, maxLength: 4)
        let values = inner.sampleBatch(key: MLX.key(0), count: 4)
        let badMask = MLXArray([Int32](arrayLiteral: 1, 0, 1, 0)).asType(.bool)
        let sample = SequenceSample(values: values, mask: badMask)
        #expect(space.containsSequence(sample) == false)
    }

    @Test
    func lengthMaskAndProbabilitySelectLengths() {
        let inner = MultiBinary(n: 2)
        let space = SequenceSpace(space: inner, minLength: 1, maxLength: 3)

        let mask = MLXArray([Int32(0), Int32(1), Int32(0)])
        let masked = space.sampleSequence(key: MLX.key(1), mask: mask, probability: nil)
        eval(masked.mask)
        #expect(masked.mask.asArray(Bool.self) == [true, true, false])

        let probability = MLXArray([Float(0), Float(0), Float(1)])
        let weighted = space.sampleSequence(
            key: MLX.key(2),
            mask: nil,
            probability: probability
        )
        eval(weighted.mask)
        #expect(weighted.mask.asArray(Bool.self) == [true, true, true])
    }
}
