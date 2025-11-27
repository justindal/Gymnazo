import Testing
import MLX

@testable import Gymnazo

@Suite("Discrete space sampling and membership")
struct DiscreteSpaceTests {
    @Test
    func testContains() async throws {
        let space = Discrete(n: 5, start: 0)
        #expect(space.contains(0))
        #expect(space.contains(4))
        #expect(space.contains(-1) == false)
        #expect(space.contains(5) == false)
        
        let shifted = Discrete(n: 3, start: 2)
        #expect(shifted.contains(1) == false)
        #expect(shifted.contains(2))
        #expect(shifted.contains(4))
        #expect(shifted.contains(5) == false)
    }
    
    @Test
    func testSampleWithMask() async throws {
        var key = MLX.key(123)
        let space = Discrete(n: 6)
        // allow only index 3
        let maskValues: [Float] = (0..<6).map { $0 == 3 ? Float(1) : Float(0) }
        let mask = MLXArray(maskValues)
        for _ in 0..<20 {
            let (k, _) = MLX.split(key: key)
            key = k
            let v = space.sample(key: key, mask: mask)
            #expect(v == 3)
        }
    }
    
    @Test
    func testSampleWithProbability() async throws {
        var key = MLX.key(456)
        let space = Discrete(n: 4)
        // all mass on index 2
        let prob = MLXArray([Float](arrayLiteral: 0.0, 0.0, 1.0, 0.0))
        for _ in 0..<20 {
            let (k, _) = MLX.split(key: key)
            key = k
            let v = space.sample(key: key, probability: prob)
            #expect(v == 2)
        }
    }
}

