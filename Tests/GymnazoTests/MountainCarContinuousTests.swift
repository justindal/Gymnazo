import Testing
import MLX
@testable import Gymnazo

@Suite("MountainCarContinuous environment")
struct MountainCarContinuousTests {
    @Test
    func testGoalVelocityParameterAffectsTermination() async throws {
        var env = MountainCarContinuous(goal_velocity: 0.04)
        _ = env.reset(seed: 0)
        
        // At goal position but too slow -> should not terminate.
        env.state = MLXArray([env.goalPosition + 0.01, 0.01] as [Float32])
        let slow = env.step(MLXArray([0.0] as [Float32]))
        #expect(slow.terminated == false)
        
        // At goal position and fast enough -> should terminate.
        env.state = MLXArray([env.goalPosition + 0.01, 0.05] as [Float32])
        let fast = env.step(MLXArray([0.0] as [Float32]))
        #expect(fast.terminated == true)
    }
    
    @Test
    @MainActor
    func testGymnazoMakeMountainCarContinuousWithKwargs() async throws {
        let env = Gymnazo.make("MountainCarContinuous", kwargs: ["goal_velocity": Float(0.04)])
        var mc = env.unwrapped as! MountainCarContinuous
        _ = mc.reset(seed: 0)
        mc.state = MLXArray([mc.goalPosition + 0.01, 0.05] as [Float32])
        let result = mc.step(MLXArray([0.0] as [Float32]))
        #expect(result.terminated == true)
    }
}


