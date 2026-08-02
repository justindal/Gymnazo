import Testing

@testable import Gymnazo

private struct CompositeValueEnvironment: ValueEnvironment {
    let valueObservationSpace: DictionaryValueSpace
    let valueActionSpace: DiscreteValueSpace
    private let observation: [String: SpaceValue]

    init() throws {
        let discrete = try DiscreteValueSpace(n: 2)
        let text = try TextValueSpace(minLength: 1, maxLength: 2, charset: ["a", "b"])
        let tuple = TupleValueSpace([AnyValueSpace(discrete), AnyValueSpace(text)])
        let sequence = try SequenceValueSpace(space: discrete, minLength: 1, maxLength: 3)
        let graph = try GraphValueSpace(
            nodeSpace: discrete,
            edgeSpace: text,
            maxNodes: 2,
            maxEdges: 1
        )
        valueObservationSpace = DictionaryValueSpace([
            "tuple": AnyValueSpace(tuple),
            "sequence": AnyValueSpace(sequence),
            "graph": AnyValueSpace(graph),
        ])
        valueActionSpace = discrete
        observation = [
            "tuple": .tuple([.discrete(1), .text("ab")]),
            "sequence": .sequence([.discrete(0), .discrete(1)]),
            "graph": .graph(
                GraphSpaceValue(
                    nodes: [.discrete(0), .discrete(1)],
                    edges: [
                        GraphSpaceEdgeValue(
                            source: 0,
                            target: 1,
                            value: .text("a")
                        )
                    ]
                )
            ),
        ]
    }

    mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<[String: SpaceValue]> {
        ValueReset(observation: observation)
    }

    mutating func stepValue(_ action: Int) throws -> ValueStep<[String: SpaceValue]> {
        ValueStep(
            observation: observation,
            reward: Double(action),
            terminated: action == 1,
            truncated: false
        )
    }

    mutating func close() {}
}

@Suite("Composite typed environment contracts")
struct CompositeValueEnvironmentTests {
    @Test
    func compositeValuesSurviveWrappersVectorsAndRegistryErasure() async throws {
        var wrapped = try CompositeValueEnvironment().checked().timeLimited(2)
        let reset = try wrapped.resetValue(seed: 3)
        #expect(wrapped.valueObservationSpace.containsValue(reset.observation))
        let step = try wrapped.stepValue(0)
        #expect(wrapped.valueObservationSpace.containsValue(step.observation))

        var vector = try SyncValueVectorEnvironment(
            environments: [
                try CompositeValueEnvironment(),
                try CompositeValueEnvironment(),
            ],
            autoresetMode: .sameStep
        )
        let vectorReset = try vector.reset(seed: 3)
        #expect(vectorReset.observations.count == 2)
        let vectorStep = try vector.step([1, 0])
        #expect(vectorStep.finalObservations[0] != nil)
        #expect(vectorStep.finalObservations[1] == nil)

        let id = "CompositeValueFixture-v0"
        await Gymnazo.registerValue(
            ValueEnvSpec(
                id: id,
                entryPoint: { _ in
                    AnyValueEnvironment(id: id, try CompositeValueEnvironment())
                }
            )
        )
        let erased = try await Gymnazo.makeValue(id)
        let erasedReset = try await erased.reset(seed: 3)
        #expect(erased.observationSpace.containsValue(erasedReset.observation))
        let erasedStep = try await erased.step(.discrete(1))
        #expect(erased.observationSpace.containsValue(erasedStep.observation))
    }
}
