import Testing
import MLX
@testable import Gymnazo

@Suite("MountainCarContinuous environment", .serialized)
struct MountainCarContinuousTests {
    func makeMountainCarContinuous(goalVelocity: Float? = nil) async throws -> MountainCarContinuous {
        let options: EnvOptions = goalVelocity.map { ["goal_velocity": $0] } ?? [:]
        let env = try await Gymnazo.make(
            "MountainCarContinuous",
            options: options
        )
        guard let mountainCar = env.unwrapped as? MountainCarContinuous else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "MountainCarContinuous",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return mountainCar
    }

    func runUntilTerminated(
        goalVelocity: Float,
        seed: UInt64,
        maxSteps: Int = 5000
    ) async throws -> (terminatedStep: Int?, terminalVelocity: Float?) {
        var env = try await makeMountainCarContinuous(goalVelocity: goalVelocity)
        var obs = try env.reset(seed: seed).obs

        for step in 1...maxSteps {
            let velocity = obs[1].item(Float.self)
            let actionValue: Float = velocity >= 0 ? 1.0 : -1.0
            let result = try env.step(MLXArray([actionValue] as [Float32]))
            obs = result.obs
            if result.terminated {
                return (step, obs[1].item(Float.self))
            }
        }

        return (nil, nil)
    }

    @Test
    func testGoalVelocityParameterAffectsTermination() async throws {
        let lowGoal = try await runUntilTerminated(
            goalVelocity: 0.0,
            seed: 0
        )
        let highGoal = try await runUntilTerminated(
            goalVelocity: 0.04,
            seed: 0
        )

        #expect(lowGoal.terminatedStep != nil)
        #expect(highGoal.terminatedStep != nil)
        #expect((lowGoal.terminatedStep ?? .max) <= (highGoal.terminatedStep ?? .max))
        #expect((highGoal.terminalVelocity ?? 0.0) >= 0.04 - 1e-4)
    }
    
    @Test
    @MainActor
    func testGymnazoMakeMountainCarContinuousWithKwargs() async throws {
        let env = try await Gymnazo.make(
            "MountainCarContinuous",
            options: ["goal_velocity": Float(0.04)]
        )
        var mc = env.unwrapped as! MountainCarContinuous
        #expect(abs(mc.goalVelocity - 0.04) < 1e-6)

        var obs = try mc.reset(seed: 0).obs
        var terminated = false
        var terminalVelocity: Float = 0.0
        for _ in 0..<5000 {
            let velocity = obs[1].item(Float.self)
            let actionValue: Float = velocity >= 0 ? 1.0 : -1.0
            let result = try mc.step(MLXArray([actionValue] as [Float32]))
            obs = result.obs
            if result.terminated {
                terminated = true
                terminalVelocity = obs[1].item(Float.self)
                break
            }
        }

        #expect(terminated)
        #expect(terminalVelocity >= mc.goalVelocity - 1e-4)
    }
}


