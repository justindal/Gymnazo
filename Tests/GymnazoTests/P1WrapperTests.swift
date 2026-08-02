import MLX
import Testing

@testable import Gymnazo

private struct TensorP1Environment: ValueEnvironment {
    let valueObservationSpace: BoxValueSpace
    let valueActionSpace: DiscreteValueSpace
    private var step = 0

    init() throws {
        valueObservationSpace = try BoxValueSpace(low: 0, high: 20, shape: [2])
        valueActionSpace = try DiscreteValueSpace(n: 2)
    }

    mutating func resetValue(
        seed _: UInt64?,
        options _: ValueResetOptions?
    ) throws -> ValueReset<TensorValue<Float>> {
        step = 0
        return ValueReset(
            observation: try TensorValue([0, 0], shape: [2])
        )
    }

    mutating func stepValue(_ action: Int) throws -> ValueStep<TensorValue<Float>> {
        step += 1
        return ValueStep(
            observation: try TensorValue([Float(step), Float(action)], shape: [2]),
            reward: 1,
            terminated: step >= 10,
            truncated: false,
            info: ["action": .discrete(action)]
        )
    }

    mutating func close() {}
}

private struct DictionaryP1Environment: ValueEnvironment {
    let valueObservationSpace: DictionaryValueSpace
    let valueActionSpace: DiscreteValueSpace

    init() throws {
        valueObservationSpace = DictionaryValueSpace([
            "keep": AnyValueSpace(try DiscreteValueSpace(n: 2)),
            "drop": AnyValueSpace(try DiscreteValueSpace(n: 3)),
        ])
        valueActionSpace = try DiscreteValueSpace(n: 1)
    }

    mutating func resetValue(
        seed _: UInt64?,
        options _: ValueResetOptions?
    ) throws -> ValueReset<[String: SpaceValue]> {
        ValueReset(observation: ["keep": .discrete(0), "drop": .discrete(1)])
    }

    mutating func stepValue(_ action: Int) throws -> ValueStep<[String: SpaceValue]> {
        ValueStep(
            observation: ["keep": .discrete(1), "drop": .discrete(2)],
            reward: Double(action),
            terminated: false,
            truncated: false
        )
    }

    mutating func close() {}
}

@Suite("P1 typed wrappers")
struct P1WrapperTests {
    @Test
    func reshapeDelayAndTimeRemainInsideDeclaredSpaces() throws {
        var reshaped = try TensorP1Environment().observationsReshaped(to: [1, 2])
        let reshapedReset = try reshaped.resetValue(seed: 1)
        #expect(reshapedReset.observation.shape == [1, 2])
        #expect(reshaped.valueObservationSpace.containsValue(reshapedReset.observation))

        var delayed = try TensorP1Environment().observationsDelayed(by: 2)
        let reset = try delayed.resetValue(seed: 2)
        let first = try delayed.stepValue(1)
        let second = try delayed.stepValue(1)
        #expect(reset.observation.elements == [0, 0])
        #expect(first.observation.elements == [0, 0])
        #expect(second.observation.elements == [0, 0])

        var timed = try TensorP1Environment().timeAware(
            maxEpisodeSteps: 10,
            normalizeTime: true
        )
        let timedReset = try timed.resetValue(seed: 3)
        let timedStep = try timed.stepValue(0)
        #expect(timedReset.observation.elements.last == 0)
        #expect(timedStep.observation.elements.last == 0.1)
        #expect(timed.valueObservationSpace.containsValue(timedStep.observation))
    }

    @Test
    func maxSkipStickyAndClipPreserveStepContracts() throws {
        var skipped = try TensorP1Environment().maxAndSkip(3)
        _ = try skipped.resetValue(seed: 4)
        let skippedStep = try skipped.stepValue(1)
        #expect(skippedStep.observation.elements == [3, 1])
        #expect(skippedStep.reward == 3)

        var sticky = try TensorP1Environment().stickyActions(probability: 1, seed: 5)
        _ = try sticky.resetValue(seed: 5)
        let first = try sticky.stepValue(0)
        let second = try sticky.stepValue(1)
        #expect(first.info["action"] == .discrete(0))
        #expect(second.info["action"] == .discrete(0))

        var clipped = try TensorP1Environment().rewardsClipped(min: -0.5, max: 0.5)
        _ = try clipped.resetValue(seed: 6)
        #expect(try clipped.stepValue(0).reward == 0.5)
    }

    @Test
    func dtypeTransformAndFilteringValidateOutputs() throws {
        let scalarSpace = try BoxValueSpace(low: 0, high: 3, shape: [1])
        var cast = try DictionaryP1Environment().observationsCasted(
            to: scalarSpace
        ) { observation in
            guard case .discrete(let value) = observation["drop"] else {
                throw GymnazoError.invalidState("Expected a discrete dictionary value.")
            }
            return try TensorValue([Float(value)], shape: [1])
        }
        #expect(try cast.resetValue(seed: 7).observation.elements == [1])

        var filtered = try DictionaryP1Environment().filteringObservation(keys: ["keep"])
        let reset = try filtered.resetValue(seed: 8)
        #expect(reset.observation == ["keep": .discrete(0)])
        #expect(filtered.valueObservationSpace.containsValue(reset.observation))
    }

    @Test
    func typedVectorFactorySupportsSyncParallelAndBatchTransforms() async throws {
        for mode in [ValueVectorMode.sync, .parallel] {
            let vector = try await Gymnazo.makeValueVec(
                "CartPole",
                numEnvs: 2,
                mode: mode,
                autoresetMode: .sameStep
            )
            let reset = try await vector.reset(seed: 9)
            #expect(reset.observations.count == 2)
            let step = try await vector.step([.discrete(0), .discrete(1)])
            #expect(step.observations.count == 2)
            #expect(step.rewards == [1, 1])

            let transformed = TransformAnyValueVectorEnvironment(
                vector,
                transformRewards: { $0.map { $0 * 2 } }
            )
            let transformedStep = try await transformed.step([.discrete(0), .discrete(1)])
            #expect(transformedStep.rewards == [2, 2])
            try await transformed.close()
        }

        let limited = try await Gymnazo.makeValueVec(
            "CartPole",
            numEnvs: 1,
            maxEpisodeSteps: 1
        )
        _ = try await limited.reset(seed: 10)
        let limitedStep = try await limited.step([.discrete(0)])
        #expect(limitedStep.truncations == [true])
        #expect(limitedStep.finalInfos[0]?[EnvInfoKey.timeLimitTruncated] == .bool(true))
        try await limited.close()
    }

    @Test
    func renderCollectionCapturesAndClearsFrames() throws {
        var collected = CartPole(renderMode: .human).collectingRenders()
        _ = try collected.reset(seed: 11)
        _ = try collected.step(MLXArray(Int32(0)))

        guard case .collection(let frames) = try collected.render() else {
            Issue.record("Expected collected render frames.")
            return
        }
        #expect(frames.count == 2)

        guard case .collection(let cleared) = try collected.render() else {
            Issue.record("Expected an empty render collection.")
            return
        }
        #expect(cleared.isEmpty)
    }

    @Test
    func registryUsesUnversionedIDsAndTypedMetadata() async throws {
        let compatibility = await Gymnazo.registry()
        let typed = await Gymnazo.valueRegistry()
        #expect(compatibility["CartPole"]?.metadata.renderFPS == 50)
        #expect(typed["FrozenLake"]?.metadata.renderModes.contains(.ansi) == true)
        #expect(compatibility["CartPole-v1"] == nil)
        #expect(typed["CartPole-v1"] == nil)
        #expect(typed["FrozenLake8x8"]?.maxEpisodeSteps == 200)
        #expect(EnvMetadata.bundled(id: "CartPole-v1") == EnvMetadata())
    }
}
