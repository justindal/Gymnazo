import MLX
import Testing

@testable import Gymnazo

@Suite("Snake environment")
struct SnakeTests {
    @Test
    func registersLegacyAndTypedEnvironments() async throws {
        let legacy = try await Gymnazo.make("Snake")
        let typed = try await Gymnazo.makeValue("Snake")

        #expect(legacy.spec?.id == "Snake")
        #expect(typed.id == "Snake")
        #expect(typed.observationSpace.flatDimension == 148)
        #expect(typed.actionSpace.flatDimension == 3)
    }

    @Test
    func resetIsDeterministicAndFoodIsUnoccupied() throws {
        var first = Snake()
        var second = Snake()

        let firstReset = try first.reset(seed: 42)
        let secondReset = try second.reset(seed: 42)

        #expect(firstReset.obs.asArray(Float.self) == secondReset.obs.asArray(Float.self))
        #expect(first.currentSnapshot == second.currentSnapshot)
        #expect(first.currentSnapshot.food != nil)
        #expect(!first.currentSnapshot.body.contains(first.currentSnapshot.food!))
    }

    @Test
    func spacesMatchFullStateContract() throws {
        var environment = Snake()
        let reset = try environment.reset(seed: 7)

        #expect(environment.observationSpace.shape == [148])
        #expect(environment.observationSpace.contains(reset.obs))
        #expect((environment.actionSpace as? Discrete)?.n == 3)
    }

    @Test
    func relativeActionsMoveInExpectedDirections() throws {
        var straight = Snake()
        var right = Snake()
        var left = Snake()
        _ = try straight.reset(seed: 3)
        _ = try right.reset(seed: 3)
        _ = try left.reset(seed: 3)

        _ = try straight.step(MLXArray(Int32(0)))
        _ = try right.step(MLXArray(Int32(1)))
        _ = try left.step(MLXArray(Int32(2)))

        #expect(straight.currentSnapshot.body.first == SnakeCell(x: 6, y: 6))
        #expect(right.currentSnapshot.body.first == SnakeCell(x: 5, y: 5))
        #expect(left.currentSnapshot.body.first == SnakeCell(x: 5, y: 7))
    }

    @Test
    func wallCollisionTerminatesEpisode() throws {
        var environment = Snake()
        _ = try environment.reset(seed: 11)

        var result: Step?
        for _ in 0..<7 {
            result = try environment.step(MLXArray(Int32(0)))
        }

        #expect(result?.terminated == true)
        #expect(result?.reward == -1)
    }

    @Test
    func rejectsInvalidActionsAndStepsBeforeReset() throws {
        var environment = Snake()

        #expect(throws: GymnazoError.self) {
            _ = try environment.step(MLXArray(Int32(0)))
        }

        _ = try environment.reset(seed: 3)
        #expect(throws: GymnazoError.self) {
            _ = try environment.step(MLXArray(Int32(3)))
        }
    }

    @Test
    func typedEnvironmentMatchesLegacyTrace() throws {
        var legacy = Snake()
        var typed = try SnakeValueEnvironment()
        let legacyReset = try legacy.reset(seed: 17)
        let typedReset = try typed.resetValue(seed: 17)

        #expect(legacyReset.obs.asArray(Float.self) == typedReset.observation.elements)

        for action in [0, 1, 2, 0, 2] {
            let legacyStep = try legacy.step(MLXArray(Int32(action)))
            let typedStep = try typed.stepValue(action)
            #expect(legacyStep.obs.asArray(Float.self) == typedStep.observation.elements)
            #expect(legacyStep.reward == typedStep.reward)
            #expect(legacyStep.terminated == typedStep.terminated)
        }
    }

    @Test
    func humanRenderReturnsSnapshot() throws {
        var environment = Snake(renderMode: .human)
        _ = try environment.reset(seed: 5)

        guard case .other(let value) = try environment.render() else {
            Issue.record("Expected a snapshot render output")
            return
        }
        #expect(value is SnakeSnapshot)
    }

    #if canImport(CoreGraphics)
        @Test
        func rgbRenderReturnsImage() throws {
            var environment = Snake(renderMode: .rgbArray)
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
