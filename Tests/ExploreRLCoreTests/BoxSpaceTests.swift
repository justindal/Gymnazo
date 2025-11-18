import Testing
import MLX
import MLXRandom
@testable import ExploreRLCore

@Suite("Box space sampling and membership")
struct BoxSpaceTests {
    @Test
    func testScalarBoundsSampleContains() async throws {
        let box = Box(low: -1.0, high: 1.0, shape: [3], dtype: .float32)
        var key = MLXRandom.key(777)
        for _ in 0..<10 {
            let (k, _) = MLXRandom.split(key: key)
            key = k
            let x = box.sample(key: key)
            #expect(box.contains(x))
        }
    }

    @Test
    func testArrayBoundsContains() async throws {
        let low = MLXArray([-1.0 as Float, 0.0 as Float, -2.0 as Float])
        let high = MLXArray([1.0 as Float, 2.0 as Float, 2.0 as Float])
        let box = Box(low: low, high: high, dtype: .float32)
        let inside = MLXArray([0.0 as Float, 1.0 as Float, -1.0 as Float])
        let outside = MLXArray([2.0 as Float, 1.0 as Float, -1.0 as Float])
        #expect(box.contains(inside))
        #expect(box.contains(outside) == false)
    }
}

