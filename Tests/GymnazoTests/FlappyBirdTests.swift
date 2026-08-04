import MLX
import Testing

@testable import Gymnazo

@Suite("Flappy Bird environment")
struct FlappyBirdTests {
    @Test
    func registersLegacyAndTypedEnvironments() async throws {
        let legacy = try await Gymnazo.make("FlappyBird")
        let typed = try await Gymnazo.makeValue("FlappyBird")

        #expect(legacy.spec?.id == "FlappyBird")
        #expect(await typed.id == "FlappyBird")
        #expect(typed.observationSpace.flatDimension == 4)
        #expect(typed.actionSpace.flatDimension == 2)
    }

    @Test
    func resetIsDeterministic() throws {
        var first = FlappyBird()
        var second = FlappyBird()

        let firstReset = try first.reset(seed: 42)
        let secondReset = try second.reset(seed: 42)

        #expect(firstReset.obs.asArray(Float.self) == secondReset.obs.asArray(Float.self))
        #expect(first.currentSnapshot == second.currentSnapshot)
    }

    @Test
    func observationAndActionSpacesMatchContract() throws {
        var environment = FlappyBird()
        let reset = try environment.reset(seed: 7)

        #expect(environment.observationSpace.shape == [4])
        #expect(environment.observationSpace.contains(reset.obs))
        #expect((environment.actionSpace as? Discrete)?.n == 2)
    }

    @Test
    func flapRaisesBirdRelativeToCoasting() throws {
        var coasting = FlappyBird()
        var flapping = FlappyBird()
        _ = try coasting.reset(seed: 9)
        _ = try flapping.reset(seed: 9)

        _ = try coasting.step(MLXArray(Int32(0)))
        _ = try flapping.step(MLXArray(Int32(1)))

        #expect(flapping.currentSnapshot.birdY > coasting.currentSnapshot.birdY)
        #expect(flapping.currentSnapshot.birdVelocity > coasting.currentSnapshot.birdVelocity)
    }

    @Test
    func coastingEventuallyTerminatesAtGround() throws {
        var environment = FlappyBird()
        _ = try environment.reset(seed: 11)

        var result: Step?
        for _ in 0..<100 {
            result = try environment.step(MLXArray(Int32(0)))
            if result?.terminated == true { break }
        }

        #expect(result?.terminated == true)
        #expect(result?.reward == -1)
    }

    @Test
    func rejectsInvalidActionsAndStepsBeforeReset() throws {
        var environment = FlappyBird()

        #expect(throws: GymnazoError.self) {
            _ = try environment.step(MLXArray(Int32(0)))
        }

        _ = try environment.reset(seed: 3)
        #expect(throws: GymnazoError.self) {
            _ = try environment.step(MLXArray(Int32(2)))
        }
    }

    @Test
    func typedEnvironmentMatchesLegacyTrace() throws {
        var legacy = FlappyBird()
        var typed = try FlappyBirdValueEnvironment()
        let legacyReset = try legacy.reset(seed: 17)
        let typedReset = try typed.resetValue(seed: 17)

        #expect(legacyReset.obs.asArray(Float.self) == typedReset.observation.elements)

        for action in [0, 1, 0, 0, 1, 0] {
            let legacyStep = try legacy.step(MLXArray(Int32(action)))
            let typedStep = try typed.stepValue(action)
            #expect(legacyStep.obs.asArray(Float.self) == typedStep.observation.elements)
            #expect(legacyStep.reward == typedStep.reward)
            #expect(legacyStep.terminated == typedStep.terminated)
        }
    }

    @Test
    func humanRenderReturnsSnapshot() throws {
        var environment = FlappyBird(renderMode: .human)
        _ = try environment.reset(seed: 5)

        guard case .other(let value) = try environment.render() else {
            Issue.record("Expected a snapshot render output")
            return
        }
        #expect(value is FlappyBirdSnapshot)
    }

    #if canImport(CoreGraphics)
        @Test
        func rgbRenderReturnsImage() throws {
            var environment = FlappyBird(renderMode: .rgbArray)
            _ = try environment.reset(seed: 5)

            guard case .rgbArray(let image) = try environment.render() else {
                Issue.record("Expected an RGB render output")
                return
            }
            #expect(image.width == 600)
            #expect(image.height == 400)
        }
    #endif
}
