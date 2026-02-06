import Testing
import MLX
@testable import Gymnazo

@Suite("Space flattening utilities")
struct FlatteningTests {
    func makeFrozenLake(isSlippery: Bool) async throws -> FrozenLake {
        let env = try await Gymnazo.make(
            "FrozenLake",
            options: ["is_slippery": isSlippery]
        )
        guard let frozenLake = env.unwrapped as? FrozenLake else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "FrozenLake",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return frozenLake
    }

    @Test
    func testDiscreteOneHotRoundTrip() async throws {
        let space = Discrete(n: 5)
        let flattened = flatten(space: space, sample: MLXArray(Int32(3)))
        guard let flat = flattened as? MLXArray else {
            Issue.record("Expected MLXArray from flatten")
            return
        }
        let restored = unflatten(space: space, flattened: flat)
        guard let restoredArr = restored as? MLXArray else {
            Issue.record("Expected MLXArray from unflatten")
            return
        }
        #expect(restoredArr.item(Int.self) == 3)
    }

    @Test
    func testDictOrderingIsDeterministic() async throws {
        let space = Dict([
            "b": Discrete(n: 2),
            "a": Discrete(n: 3),
        ])
        let sample: [String: Any] = [
            "a": MLXArray(Int32(1)),
            "b": MLXArray(Int32(0)),
        ]
        let flattened = flatten(space: space, sample: sample)
        guard let flat = flattened as? MLXArray else {
            Issue.record("Expected MLXArray from flatten")
            return
        }
        let restored = unflatten(space: space, flattened: flat)
        let dict = restored as? [String: Any]
        #expect((dict?["a"] as? MLXArray)?.item(Int.self) == 1)
        #expect((dict?["b"] as? MLXArray)?.item(Int.self) == 0)
    }

    @Test
    func testTupleRoundTrip() async throws {
        let space = Tuple(Discrete(n: 2), Discrete(n: 3))
        let sample: [Any] = [MLXArray(Int32(1)), MLXArray(Int32(2))]
        let flattened = flatten(space: space, sample: sample)
        guard let flat = flattened as? MLXArray else {
            Issue.record("Expected MLXArray from flatten")
            return
        }
        let restored = unflatten(space: space, flattened: flat)
        let tuple = restored as? [Any]
        #expect((tuple?[0] as? MLXArray)?.item(Int.self) == 1)
        #expect((tuple?[1] as? MLXArray)?.item(Int.self) == 2)
    }

    @Test
    func testFlattenObservationFrozenLakeIsOneHot() async throws {
        let baseEnv = try await makeFrozenLake(isSlippery: false)
        var env = try baseEnv.observationsFlattened()
        let obs = try env.reset(seed: 0, options: nil).obs
        eval(obs)
        let values = obs.asArray(Float.self)
        let sum = values.reduce(0, +)
        #expect(abs(sum - 1) < 1e-5)
        #expect(values.count == 16)
    }

    @Test
    func testFlattenSpaceDiscreteHasZeroOneBounds() async throws {
        let space = Discrete(n: 5)
        let flattened = flatten_space(space)
        guard let box = flattened as? Box else {
            Issue.record("Expected Box from flatten_space(Discrete)")
            return
        }
        eval(box.low, box.high)
        let low = box.low.asArray(Float.self)
        let high = box.high.asArray(Float.self)
        #expect(low.allSatisfy { $0 == 0 })
        #expect(high.allSatisfy { $0 == 1 })
        #expect(box.shape == [5])
    }

    @Test
    func testFlattenSpaceMultiDiscreteHasZeroOneBounds() async throws {
        let space = MultiDiscrete([2, 3])
        let flattened = flatten_space(space)
        guard let box = flattened as? Box else {
            Issue.record("Expected Box from flatten_space(MultiDiscrete)")
            return
        }
        eval(box.low, box.high)
        let low = box.low.asArray(Float.self)
        let high = box.high.asArray(Float.self)
        #expect(low.allSatisfy { $0 == 0 })
        #expect(high.allSatisfy { $0 == 1 })
        #expect(box.shape == [5])
    }

    @Test
    func testFlattenSpaceTextSpaceHasMinusOnePaddingBounds() async throws {
        let space = TextSpace(minLength: 0, maxLength: 4)
        let flattened = flatten_space(space)
        guard let box = flattened as? Box else {
            Issue.record("Expected Box from flatten_space(TextSpace)")
            return
        }
        eval(box.low, box.high)
        let low = box.low.asArray(Float.self)
        let high = box.high.asArray(Float.self)
        #expect(low.allSatisfy { $0 == -1 })
        #expect(high.allSatisfy { $0 == Float(space.charset.count - 1) })
        #expect(box.shape == [4])
    }

    @Test
    func testFlattenSpaceSequenceStaysSequence() async throws {
        let inner = MultiDiscrete([2, 3])
        let seq = SequenceSpace(space: inner, minLength: 0, maxLength: 7)
        let flattened = flatten_space(seq)
        #expect((flattened as? Box) == nil)
        guard let seqFlat = flattened as? any AnySequenceSpace else {
            Issue.record("Expected SequenceSpace from flatten_space(SequenceSpace)")
            return
        }
        let elementFlattened = flatten_space(seqFlat.elementSpace)
        #expect(elementFlattened is Box)
    }

    @Test
    func testFlattenSpaceGraphStaysGraph() async throws {
        let nodeSpace = MultiDiscrete([2, 3])
        let edgeSpace = MultiBinary(n: 2)
        let graph = Graph(nodeSpace: nodeSpace, edgeSpace: edgeSpace, maxNodes: 3, maxEdges: 4)
        let flattened = flatten_space(graph)
        #expect((flattened as? Box) == nil)
        guard let graphFlat = flattened as? any AnyGraphSpace else {
            Issue.record("Expected Graph from flatten_space(Graph)")
            return
        }
        #expect(flatten_space(graphFlat.nodeSpaceAny) is Box)
        #expect(flatten_space(graphFlat.edgeSpaceAny) is Box)
    }

    @Test
    func testFlattenSpaceBoxFlattensBounds() async throws {
        let low = MLXArray([Float](arrayLiteral: -1, 0)).asType(.float32)
        let high = MLXArray([Float](arrayLiteral: 1, 2)).asType(.float32)
        let space = Box(low: low, high: high, dtype: .float32)
        let flattened = flatten_space(space)
        guard let box = flattened as? Box else {
            Issue.record("Expected Box from flatten_space(Box)")
            return
        }
        eval(box.low, box.high)
        let l = box.low.asArray(Float.self)
        let h = box.high.asArray(Float.self)
        #expect(l == [-1, 0])
        #expect(h == [1, 2])
        #expect(box.shape == [2])
    }

    @Test
    func testFlattenSpaceToBoxIsNilForSequenceAndGraph() async throws {
        let inner = MultiDiscrete([2, 3])
        let seq = SequenceSpace(space: inner, minLength: 0, maxLength: 7)
        #expect(flattenSpaceToBox(seq) == nil)

        let nodeSpace = MultiDiscrete([2, 3])
        let edgeSpace = MultiBinary(n: 2)
        let graph = Graph(nodeSpace: nodeSpace, edgeSpace: edgeSpace, maxNodes: 3, maxEdges: 4)
        #expect(flattenSpaceToBox(graph) == nil)
    }
}
