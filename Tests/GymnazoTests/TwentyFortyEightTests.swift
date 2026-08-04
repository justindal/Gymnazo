import MLX
import Testing

@testable import Gymnazo

@Suite("2048 environment")
struct TwentyFortyEightTests {
    @Test
    func registersLegacyAndTypedEnvironments() async throws {
        let legacy = try await Gymnazo.make("2048")
        let typed = try await Gymnazo.makeValue("2048")

        #expect(legacy.spec?.id == "2048")
        #expect(typed.id == "2048")
        #expect(typed.observationSpace.flatDimension == 16)
        #expect(typed.actionSpace.flatDimension == 4)
    }

    @Test
    func resetIsDeterministicAndStartsWithTwoTiles() throws {
        var first = TwentyFortyEight()
        var second = TwentyFortyEight()

        let firstReset = try first.reset(seed: 42)
        let secondReset = try second.reset(seed: 42)

        #expect(firstReset.obs.asArray(Float.self) == secondReset.obs.asArray(Float.self))
        #expect(first.currentSnapshot == second.currentSnapshot)
        #expect(first.currentSnapshot.tiles.filter { $0 != 0 }.count == 2)
        #expect(first.currentSnapshot.tiles.allSatisfy { $0 == 0 || $0 == 2 || $0 == 4 })
    }

    @Test
    func spacesMatchFullStateContract() throws {
        var environment = TwentyFortyEight()
        let reset = try environment.reset(seed: 7)

        #expect(environment.observationSpace.shape == [16])
        #expect(environment.observationSpace.contains(reset.obs))
        #expect((environment.actionSpace as? Discrete)?.n == 4)
    }

    @Test
    func typedEnvironmentMatchesLegacyTrace() throws {
        var legacy = TwentyFortyEight()
        var typed = try TwentyFortyEightValueEnvironment()
        let legacyReset = try legacy.reset(seed: 17)
        let typedReset = try typed.resetValue(seed: 17)

        #expect(legacyReset.obs.asArray(Float.self) == typedReset.observation.elements)

        for action in [0, 1, 2, 3, 0, 3, 1] {
            let legacyStep = try legacy.step(MLXArray(Int32(action)))
            let typedStep = try typed.stepValue(action)
            #expect(legacyStep.obs.asArray(Float.self) == typedStep.observation.elements)
            #expect(legacyStep.reward == typedStep.reward)
            #expect(legacyStep.terminated == typedStep.terminated)
        }
    }

    @Test
    func rejectsInvalidActionsAndStepsBeforeReset() throws {
        var environment = TwentyFortyEight()

        #expect(throws: GymnazoError.self) {
            _ = try environment.step(MLXArray(Int32(0)))
        }

        _ = try environment.reset(seed: 3)
        #expect(throws: GymnazoError.self) {
            _ = try environment.step(MLXArray(Int32(4)))
        }
    }

    @Test
    func rejectsMalformedLegacyActions() throws {
        var environment = TwentyFortyEight()
        _ = try environment.reset(seed: 3)

        #expect(throws: GymnazoError.self) {
            _ = try environment.step(MLXArray([Int32(0), Int32(1)]))
        }
        #expect(throws: GymnazoError.self) {
            _ = try environment.step(MLXArray([0.0 as Float]))
        }
    }

    @Test
    func blockedDirectionIsAValidNoOp() throws {
        var environment = TwentyFortyEight()
        var foundNoOp = false

        for seed in 0..<100 {
            _ = try environment.reset(seed: UInt64(seed))
            let before = environment.currentSnapshot

            for action in 0..<4 {
                var candidate = environment
                let step = try candidate.step(MLXArray(Int32(action)))
                if candidate.currentSnapshot.tiles == before.tiles {
                    #expect(step.reward == 0)
                    #expect(!step.terminated)
                    foundNoOp = true
                    break
                }
            }

            if foundNoOp {
                break
            }
        }

        #expect(foundNoOp)
    }

    @Test
    func humanRenderReturnsSnapshot() throws {
        var environment = TwentyFortyEight(renderMode: .human)
        _ = try environment.reset(seed: 5)

        guard case .other(let value) = try environment.render() else {
            Issue.record("Expected a snapshot render output")
            return
        }
        #expect(value is TwentyFortyEightSnapshot)
    }

    #if canImport(CoreGraphics)
        @Test
        func rgbRenderReturnsSquareImage() throws {
            var environment = TwentyFortyEight(renderMode: .rgbArray)
            _ = try environment.reset(seed: 5)

            guard case .rgbArray(let image) = try environment.render() else {
                Issue.record("Expected an RGB render output")
                return
            }
            #expect(image.width == 600)
            #expect(image.height == 600)
        }
    #endif
}
