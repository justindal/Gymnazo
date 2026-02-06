import Foundation
import MLX
import Testing

@testable import Gymnazo

@Suite("MountainCar environment")
struct MountainCarTests {
    func makeMountainCar(renderMode: RenderMode? = nil, goalVelocity: Float? = nil) async throws -> MountainCar {
        var options: EnvOptions = [:]
        if let renderMode {
            options["render_mode"] = renderMode.rawValue
        }
        if let goalVelocity {
            options["goal_velocity"] = goalVelocity
        }
        let env = try await Gymnazo.make("MountainCar", options: options)
        guard let mountainCar = env.unwrapped as? MountainCar else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "MountainCar",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return mountainCar
    }

    @Test
    func testInitialization() async throws {
        let env = try await makeMountainCar()

        #expect(env.minPosition == -1.2)
        #expect(env.maxPosition == 0.6)
        #expect(env.maxSpeed == 0.07)
        #expect(env.goalPosition == 0.5)
        #expect(env.force == 0.001)
        #expect(env.gravity == 0.0025)
        #expect(env.state == nil)
    }

    @Test
    func testActionSpace() async throws {
        let env = try await makeMountainCar()

        guard let actionSpace = env.actionSpace as? Discrete else {
            Issue.record("Action space is not Discrete")
            return
        }

        #expect(actionSpace.n == 3)
        #expect(actionSpace.start == 0)
    }

    @Test
    func testObservationSpace() async throws {
        let env = try await makeMountainCar()

        #expect(env.observationSpace.shape == [2])
    }

    @Test
    func testRenderModeInitialization() async throws {
        let envNoRender = try await makeMountainCar(renderMode: nil)
        #expect(envNoRender.renderMode == nil)

        let envHuman = try await makeMountainCar(renderMode: .human)
        #expect(envHuman.renderMode == .human)
    }

    @Test
    func testReset() async throws {
        var env = try await makeMountainCar()
        let result = try env.reset()

        #expect(result.obs.shape == [2])
        #expect(env.state != nil)

        let position = result.obs[0].item(Float.self)
        let velocity = result.obs[1].item(Float.self)

        #expect(position >= -0.6)
        #expect(position <= -0.4)

        #expect(abs(velocity) < 0.0001)
    }

    @Test
    func testResetWithSeed() async throws {
        var env1 = try await makeMountainCar()
        var env2 = try await makeMountainCar()

        let result1 = try env1.reset(seed: 42)
        let result2 = try env2.reset(seed: 42)

        let pos1 = result1.obs[0].item(Float.self)
        let pos2 = result2.obs[0].item(Float.self)

        #expect(abs(pos1 - pos2) < 0.0001, "Same seed should produce same initial position")
    }

    @Test
    func testResetDifferentSeeds() async throws {
        var env1 = try await makeMountainCar()
        var env2 = try await makeMountainCar()

        let result1 = try env1.reset(seed: 42)
        let result2 = try env2.reset(seed: 123)

        let pos1 = result1.obs[0].item(Float.self)
        let pos2 = result2.obs[0].item(Float.self)

        #expect(pos1 != pos2, "Different seeds should produce different positions")
    }

    @Test
    func testStepShape() async throws {
        var env = try await makeMountainCar()
        _ = try env.reset()

        let result = try env.step(MLXArray(Int32(1)))

        #expect(result.obs.shape == [2])
        #expect(result.reward == -1.0)
        #expect(result.truncated == false)
    }

    @Test
    func testStepAllActions() async throws {
        var env = try await makeMountainCar()

        for action in 0..<3 {
            _ = try env.reset()
            let result = try env.step(MLXArray(Int32(action)))
            #expect(result.obs.shape == [2], "Action \(action) should return valid observation")
        }
    }

    @Test
    func testStepPhysicsPushRight() async throws {
        var env = try await makeMountainCar()
        _ = try env.reset(seed: 42)

        let initialVel = env.state![1].item(Float.self)

        for _ in 0..<10 {
            _ = try env.step(MLXArray(Int32(2)))
        }

        let finalVel = env.state![1].item(Float.self)

        #expect(finalVel != initialVel, "Velocity should change after pushing")
    }

    @Test
    func testStepPhysicsPushLeft() async throws {
        var env = try await makeMountainCar()
        _ = try env.reset(seed: 42)

        let initialVel = env.state![1].item(Float.self)

        for _ in 0..<10 {
            _ = try env.step(MLXArray(Int32(0)))
        }

        let finalVel = env.state![1].item(Float.self)

        #expect(finalVel != initialVel, "Velocity should change after pushing")
    }

    @Test
    func testPositionBounds() async throws {
        var env = try await makeMountainCar()
        _ = try env.reset()

        for _ in 0..<500 {
            let result = try env.step(MLXArray(Int32(0)))
            let position = result.obs[0].item(Float.self)

            #expect(position >= env.minPosition, "Position should not go below min")
            #expect(position <= env.maxPosition, "Position should not exceed max")

            if result.terminated { break }
        }
    }

    @Test
    func testVelocityBounds() async throws {
        var env = try await makeMountainCar()
        _ = try env.reset()

        for _ in 0..<500 {
            let result = try env.step(MLXArray(Int32(2)))
            let velocity = result.obs[1].item(Float.self)

            #expect(velocity >= -env.maxSpeed, "Velocity should not go below -maxSpeed")
            #expect(velocity <= env.maxSpeed, "Velocity should not exceed maxSpeed")

            if result.terminated { break }
        }
    }

    @Test
    func testLeftBoundaryStopsVelocity() async throws {
        var env = try await makeMountainCar()
        _ = try env.reset()

        env.state = MLXArray([-1.15, -0.05] as [Float32])

        var hitBoundary = false
        for _ in 0..<100 {
            let result = try env.step(MLXArray(Int32(0)))
            let position = result.obs[0].item(Float.self)
            let velocity = result.obs[1].item(Float.self)

            if position == env.minPosition {
                #expect(velocity >= 0, "At left boundary, velocity should be stopped (>= 0)")
                hitBoundary = true
                break
            }

            if result.terminated { break }
        }

        #expect(hitBoundary, "Should have hit the left boundary")
    }

    @Test
    func testTerminationCondition() async throws {
        var env = try await makeMountainCar()
        _ = try env.reset()

        env.state = MLXArray([env.goalPosition + 0.01, 0.01])

        let result = try env.step(MLXArray(Int32(1)))

        #expect(
            result.terminated == true,
            "Should terminate when at goal position with non-negative velocity")
    }

    @Test
    func testGoalVelocityParameterAffectsTermination() async throws {
        var env = try await makeMountainCar(goalVelocity: 0.04)
        _ = try env.reset()

        env.state = MLXArray([env.goalPosition + 0.01, 0.01] as [Float32])
        let slow = try env.step(MLXArray(Int32(1)))
        #expect(slow.terminated == false)

        env.state = MLXArray([env.goalPosition + 0.01, 0.05] as [Float32])
        let fast = try env.step(MLXArray(Int32(1)))
        #expect(fast.terminated == true)
    }

    @Test
    func testNoTerminationBeforeGoal() async throws {
        var env = try await makeMountainCar()
        _ = try env.reset()

        env.state = MLXArray([env.goalPosition - 0.1, 0.0])

        let result = try env.step(MLXArray(Int32(1)))

        #expect(result.terminated == false, "Should not terminate before reaching goal")
    }

    @Test
    func testHeightFunctionSinusoidal() async throws {
        let position: Float = 0.0
        let expectedHeight: Float = Foundation.sin(3.0 * position) * 0.45 + 0.55
        let actualHeight = MountainCar.height(at: position)

        #expect(abs(actualHeight - expectedHeight) < 0.0001)
    }

    @Test
    func testHeightFunctionValley() async throws {
        let heightAtValley = MountainCar.height(at: -0.5)
        let heightAtLeft = MountainCar.height(at: -1.2)
        let heightAtRight = MountainCar.height(at: 0.6)

        #expect(heightAtValley < heightAtLeft, "Valley should be lower than left edge")
        #expect(heightAtValley < heightAtRight, "Valley should be lower than right edge")
    }

    @Test
    func testSnapshotCreation() async throws {
        var env = try await makeMountainCar()
        _ = try env.reset()

        let snapshot = env.currentSnapshot

        #expect(snapshot.position >= env.minPosition)
        #expect(snapshot.position <= env.maxPosition)
        #expect(abs(snapshot.velocity) < 0.0001)
        #expect(snapshot.minPosition == env.minPosition)
        #expect(snapshot.maxPosition == env.maxPosition)
        #expect(snapshot.goalPosition == env.goalPosition)
    }

    @Test
    func testSnapshotHeight() async throws {
        let snapshot = MountainCarSnapshot(
            position: -0.5,
            velocity: 0,
            minPosition: -1.2,
            maxPosition: 0.6,
            goalPosition: 0.5
        )

        let expectedHeight = MountainCar.height(at: -0.5)
        #expect(abs(snapshot.height - expectedHeight) < 0.0001)
    }

    @Test
    func testSnapshotZero() async throws {
        let zero = MountainCarSnapshot.zero

        #expect(zero.position == -0.5)
        #expect(zero.velocity == 0)
        #expect(zero.minPosition == -1.2)
        #expect(zero.maxPosition == 0.6)
        #expect(zero.goalPosition == 0.5)
    }

    @Test @MainActor
    func testGymnazoRegistration() async throws {
        var env = try await Gymnazo.make("MountainCar")
        let result = try env.reset()
        let obs = result.obs
        #expect(obs.shape == [2])
    }

    @Test @MainActor
    func testGymnazoMaxEpisodeSteps() async throws {
        let env = try await Gymnazo.make("MountainCar")
        #expect(env.spec?.maxEpisodeSteps == 200)
    }
}
