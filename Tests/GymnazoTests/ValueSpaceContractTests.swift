import Testing

@testable import Gymnazo

@Suite("Value space contracts")
struct ValueSpaceContractTests {
    @Test
    func tensorValueValidatesShape() throws {
        #expect(throws: GymnazoError.self) {
            _ = try TensorValue([1.0, 2.0], shape: [3])
        }

        let tensor = try TensorValue([1.0, 2.0], shape: [2])
        #expect(tensor.count == 2)
        #expect(tensor.shape == [2])
    }

    @Test
    func primitiveSpacesSampleContainedValuesDeterministically() throws {
        let discrete = try DiscreteValueSpace(n: 3, start: -1)
        let box = try BoxValueSpace(low: -1, high: 1, shape: [2])
        let multiBinary = try MultiBinaryValueSpace(n: 4)
        let multiDiscrete = try MultiDiscreteValueSpace([2, 3], start: [1, -2])
        let text = try TextValueSpace(minLength: 2, maxLength: 5, charset: ["a", "b", "c"])

        var first = SeededGenerator(seed: 42)
        var second = SeededGenerator(seed: 42)

        let samples = try [
            SpaceValue.discrete(discrete.sample(using: &first)),
            .floatTensor(box.sample(using: &first)),
            .boolTensor(multiBinary.sample(using: &first)),
            .intTensor(multiDiscrete.sample(using: &first)),
            .text(text.sample(using: &first)),
        ]
        let repeated = try [
            SpaceValue.discrete(discrete.sample(using: &second)),
            .floatTensor(box.sample(using: &second)),
            .boolTensor(multiBinary.sample(using: &second)),
            .intTensor(multiDiscrete.sample(using: &second)),
            .text(text.sample(using: &second)),
        ]

        #expect(samples == repeated)
        #expect(discrete.containsValue(try discrete.sample(using: &first)))
        #expect(box.containsValue(try box.sample(using: &first)))
        #expect(multiBinary.containsValue(try multiBinary.sample(using: &first)))
        #expect(multiDiscrete.containsValue(try multiDiscrete.sample(using: &first)))
        #expect(text.containsValue(try text.sample(using: &first)))
    }

    @Test
    func batchingRoundTripsAndRejectsInvalidValues() throws {
        let space = try DiscreteValueSpace(n: 2)
        let batch = try space.batch([0, 1])

        #expect(batch.count == 2)
        #expect(try space.unbatch(batch) == [0, 1])

        #expect(throws: GymnazoError.self) {
            _ = try space.batch([0, 2])
        }
    }

    @Test
    func compositeSpacesPreserveNativeValuesAndSeededSampling() throws {
        let discrete = try DiscreteValueSpace(n: 3)
        let text = try TextValueSpace(minLength: 1, maxLength: 3, charset: ["a", "b"])
        let tuple = TupleValueSpace([AnyValueSpace(discrete), AnyValueSpace(text)])
        let dictionary = DictionaryValueSpace([
            "action": AnyValueSpace(discrete),
            "label": AnyValueSpace(text),
        ])
        let sequence = try SequenceValueSpace(space: discrete, minLength: 1, maxLength: 4)
        let graph = try GraphValueSpace(
            nodeSpace: discrete,
            edgeSpace: text,
            maxNodes: 4,
            maxEdges: 5,
            allowSelfLoops: false
        )

        var first = SeededGenerator(seed: 99)
        var second = SeededGenerator(seed: 99)

        let tupleValue = try tuple.sample(using: &first)
        let dictionaryValue = try dictionary.sample(using: &first)
        let sequenceValue = try sequence.sample(using: &first)
        let graphValue = try graph.sample(using: &first)

        #expect(tuple.containsValue(tupleValue))
        #expect(dictionary.containsValue(dictionaryValue))
        #expect(sequence.containsValue(sequenceValue))
        #expect(graph.containsValue(graphValue))

        #expect(
            tuple.encodeValue(tupleValue) == tuple.encodeValue(try tuple.sample(using: &second)))
        #expect(
            dictionary.encodeValue(dictionaryValue)
                == dictionary.encodeValue(try dictionary.sample(using: &second))
        )
        #expect(
            sequence.encodeValue(sequenceValue)
                == sequence.encodeValue(try sequence.sample(using: &second))
        )
        #expect(
            graph.encodeValue(graphValue)
                == graph.encodeValue(try graph.sample(using: &second))
        )
    }

    @Test
    func flatteningRoundTripsFixedAndStructuralSpaces() throws {
        let discrete = try DiscreteValueSpace(n: 3, start: -1)
        let multiDiscrete = try MultiDiscreteValueSpace([2, 3], start: [1, -2])
        let text = try TextValueSpace(minLength: 1, maxLength: 3, charset: ["a", "b"])
        let tuple = TupleValueSpace([AnyValueSpace(discrete), AnyValueSpace(text)])
        let dictionary = DictionaryValueSpace([
            "choice": AnyValueSpace(discrete),
            "modes": AnyValueSpace(multiDiscrete),
        ])
        let sequence = try SequenceValueSpace(space: discrete, minLength: 1, maxLength: 4)
        let graph = try GraphValueSpace(
            nodeSpace: discrete,
            edgeSpace: text,
            maxNodes: 3,
            maxEdges: 4
        )

        let tupleValue: [SpaceValue] = [.discrete(1), .text("ab")]
        let dictionaryValue: [String: SpaceValue] = [
            "choice": .discrete(0),
            "modes": .intTensor(try TensorValue([2, 0], shape: [2])),
        ]
        let sequenceValue = [-1, 0, 1]
        let graphValue = GraphValue(
            nodes: [-1, 1],
            edges: [GraphEdge(source: 0, target: 1, value: "ba")]
        )

        #expect(try tuple.unflattenValue(tuple.flattenValue(tupleValue)) == tupleValue)
        #expect(
            try dictionary.unflattenValue(dictionary.flattenValue(dictionaryValue))
                == dictionaryValue
        )
        #expect(try sequence.unflattenValue(sequence.flattenValue(sequenceValue)) == sequenceValue)
        #expect(
            graph.encodeValue(try graph.unflattenValue(graph.flattenValue(graphValue)))
                == graph.encodeValue(graphValue)
        )

        guard case .vector(let tupleVector) = try tuple.flattenValue(tupleValue) else {
            Issue.record("Expected a fixed tuple to flatten to one vector.")
            return
        }
        #expect(tupleVector.count == 6)

        guard case .sequence = try sequence.flattenValue(sequenceValue) else {
            Issue.record("Expected a sequence to preserve variable structure.")
            return
        }
        guard case .graph = try graph.flattenValue(graphValue) else {
            Issue.record("Expected a graph to preserve graph structure.")
            return
        }
    }

    @Test
    func mlxCodecBatchesAtOneBoundary() throws {
        let first = try TensorValue([1.0, 2.0] as [Float], shape: [2])
        let second = try TensorValue([3.0, 4.0] as [Float], shape: [2])

        let batch = try MLXValueCodec.encodeBatch([first, second])
        let decoded = try MLXValueCodec.decodeFloat(batch)

        let expected = try TensorValue([1.0, 2.0, 3.0, 4.0] as [Float], shape: [2, 2])
        #expect(decoded == expected)
    }

    @Test
    func textFlatteningRejectsCharactersAfterPadding() throws {
        let text = try TextValueSpace(
            minLength: 1,
            maxLength: 3,
            charset: ["a", "b"]
        )

        #expect(throws: GymnazoError.self) {
            try text.unflattenValue(.vector([0, 2, 1]))
        }
    }
}
