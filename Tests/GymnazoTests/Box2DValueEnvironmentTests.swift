import MLX
import Testing

@testable import Gymnazo

@Suite("Box2D typed environment boundaries")
struct Box2DValueEnvironmentTests {
    @Test
    func lunarLanderMatchesCompatibilityTrace() async throws {
        var compatibility = try LunarLander()
        let typed = try await Gymnazo.makeValue("LunarLander")

        let compatibilityReset = try compatibility.reset(seed: 41)
        let typedReset = try await typed.reset(seed: 41)
        expectPixelsOrFloats(compatibilityReset.obs, typedReset.observation)

        for action in [0, 2, 1, 3] {
            let compatibilityStep = try compatibility.step(MLXArray(Int32(action)))
            let typedStep = try await typed.step(.discrete(action))
            expectPixelsOrFloats(compatibilityStep.obs, typedStep.observation)
            #expect(compatibilityStep.reward == typedStep.reward)
            #expect(compatibilityStep.terminated == typedStep.terminated)
            #expect(compatibilityStep.truncated == typedStep.truncated)
            if compatibilityStep.terminated || compatibilityStep.truncated { break }
        }
    }

    @Test
    func lunarLanderContinuousMatchesCompatibilityTrace() async throws {
        var compatibility = try LunarLanderContinuous()
        let typed = try await Gymnazo.makeValue("LunarLanderContinuous")

        let compatibilityReset = try compatibility.reset(seed: 73)
        let typedReset = try await typed.reset(seed: 73)
        expectPixelsOrFloats(compatibilityReset.obs, typedReset.observation)

        for action in [[0, 0], [1, 0], [0, -1], [-1, 1]] as [[Float]] {
            let compatibilityStep = try compatibility.step(MLXArray(action))
            let typedStep = try await typed.step(
                .floatTensor(try TensorValue(action, shape: [2]))
            )
            expectPixelsOrFloats(compatibilityStep.obs, typedStep.observation)
            #expect(compatibilityStep.reward == typedStep.reward)
            #expect(compatibilityStep.terminated == typedStep.terminated)
            #expect(compatibilityStep.truncated == typedStep.truncated)
            if compatibilityStep.terminated || compatibilityStep.truncated { break }
        }
    }

    @Test
    func carRacingUsesTypedPixelAndActionBoundaries() async throws {
        let continuous = try await Gymnazo.makeValue("CarRacing")
        let continuousReset = try await continuous.reset(seed: 29)
        guard case .floatTensor(let pixels) = continuousReset.observation else {
            Issue.record("Expected typed CarRacing pixels.")
            return
        }
        #expect(pixels.shape == [96, 96, 3])
        #expect(pixels.elements.allSatisfy { (0...255).contains($0) })
        #expect(pixels.elements.contains { $0 != 128 })
        let continuousStep = try await continuous.step(
            .floatTensor(try TensorValue([0, 1, 0], shape: [3]))
        )
        #expect(continuous.observationSpace.containsValue(continuousStep.observation))

        let discrete = try await Gymnazo.makeValue("CarRacingDiscrete")
        let discreteReset = try await discrete.reset(
            seed: 29,
            options: ["randomize": .bool(false)]
        )
        #expect(discrete.observationSpace.containsValue(discreteReset.observation))
        let discreteStep = try await discrete.step(.discrete(3))
        #expect(discrete.observationSpace.containsValue(discreteStep.observation))
    }

    private func expectPixelsOrFloats(_ compatibility: MLXArray, _ typed: SpaceValue) {
        guard case .floatTensor(let value) = typed else {
            Issue.record("Expected a typed float tensor.")
            return
        }
        let expected = compatibility.asType(.float32).asArray(Float.self)
        #expect(value.shape == compatibility.shape)
        #expect(value.elements == expected)
    }
}
