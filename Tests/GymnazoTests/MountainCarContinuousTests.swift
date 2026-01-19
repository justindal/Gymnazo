import Testing
import MLX
@testable import Gymnazo

@Suite("MountainCarContinuous environment")
struct MountainCarContinuousTests {
    func makeMountainCarContinuous(goalVelocity: Float? = nil) async throws -> MountainCarContinuous {
        let options: EnvOptions = goalVelocity.map { ["goal_velocity": $0] } ?? [:]
        let env: AnyEnv<MLXArray, MLXArray> = try await Gymnazo.make(
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

    @Test
    func testGoalVelocityParameterAffectsTermination() async throws {
        var env = try await makeMountainCarContinuous(goalVelocity: 0.04)
        _ = try env.reset(seed: 0)
        
        env.state = MLXArray([env.goalPosition + 0.01, 0.01] as [Float32])
        let slow = try env.step(MLXArray([0.0] as [Float32]))
        #expect(slow.terminated == false)
        
        env.state = MLXArray([env.goalPosition + 0.01, 0.05] as [Float32])
        let fast = try env.step(MLXArray([0.0] as [Float32]))
        #expect(fast.terminated == true)
    }
    
    @Test
    @MainActor
    func testGymnazoMakeMountainCarContinuousWithKwargs() async throws {
        let env: AnyEnv<MLXArray, MLXArray> = try await Gymnazo.make(
            "MountainCarContinuous",
            options: ["goal_velocity": Float(0.04)]
        )
        var mc = env.unwrapped as! MountainCarContinuous
        _ = try mc.reset(seed: 0)
        mc.state = MLXArray([mc.goalPosition + 0.01, 0.05] as [Float32])
        let result = try mc.step(MLXArray([0.0] as [Float32]))
        #expect(result.terminated == true)
    }
}


