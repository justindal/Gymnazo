import MLX
import Testing

@testable import Gymnazo

@Suite("MultiDiscrete space sampling and membership")
struct MultiDiscreteSpaceTests {
    @Test
    func startOffsetsDefineMembershipAndSampling() {
        let space = MultiDiscrete([2, 3], start: [4, -2])

        #expect(space.contains(MLXArray([Int32(4), Int32(-2)])))
        #expect(space.contains(MLXArray([Int32(5), Int32(0)])))
        #expect(!space.contains(MLXArray([Int32(3), Int32(-2)])))
        #expect(!space.contains(MLXArray([Int32(4), Int32(1)])))
        #expect(!space.contains(MLXArray([Float(4), Float(-2)])))

        let samples = space.sampleBatch(key: MLX.key(17), count: 64)
        eval(samples)
        let values = samples.asArray(Int32.self)
        for row in 0..<64 {
            #expect((4...5).contains(values[row * 2]))
            #expect((-2...0).contains(values[row * 2 + 1]))
        }
    }

    @Test
    func maskAndProbabilityRespectPaddedRowsAndStart() {
        let space = MultiDiscrete([2, 3], start: [4, -2])
        let mask = MLXArray([
            Int32(0), Int32(1), Int32(0),
            Int32(1), Int32(0), Int32(0),
        ]).reshaped([2, 3])
        let masked = space.sample(key: MLX.key(3), mask: mask, probability: nil)
        eval(masked)
        #expect(masked.asArray(Int32.self) == [5, -2])

        let probability = MLXArray([
            Float(1), Float(0), Float(0),
            Float(0), Float(0), Float(1),
        ]).reshaped([2, 3])
        let weighted = space.sample(key: MLX.key(4), mask: nil, probability: probability)
        eval(weighted)
        #expect(weighted.asArray(Int32.self) == [4, 0])
    }
}
