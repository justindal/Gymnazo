//
//  MountainCarTests.swift
//

import Testing
import MLX
import Foundation
@testable import ExploreRLCore

@Suite("MountainCar environment")
struct MountainCarTests {
    
    @Test
    func testInitialization() async throws {
        let env = MountainCar()
        
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
        let env = MountainCar()
        
        // MountainCar has 3 actions: push left (0), no push (1), push right (2)
        #expect(env.action_space.n == 3)
        #expect(env.action_space.start == 0)
    }
    
    @Test
    func testObservationSpace() async throws {
        let env = MountainCar()
        
        // Observation: [position, velocity]
        #expect(env.observation_space.shape == [2])
    }
    
    @Test
    func testRenderModeInitialization() async throws {
        let envNoRender = MountainCar(render_mode: nil)
        #expect(envNoRender.render_mode == nil)
        
        let envHuman = MountainCar(render_mode: "human")
        #expect(envHuman.render_mode == "human")
    }
    
    @Test
    func testReset() async throws {
        var env = MountainCar()
        let result = env.reset()
        
        #expect(result.obs.shape == [2])
        #expect(env.state != nil)
        
        let position = result.obs[0].item(Float.self)
        let velocity = result.obs[1].item(Float.self)
        
        // Position should be in [-0.6, -0.4]
        #expect(position >= -0.6)
        #expect(position <= -0.4)
        
        // Velocity should be 0
        #expect(abs(velocity) < 0.0001)
    }
    
    @Test
    func testResetWithSeed() async throws {
        var env1 = MountainCar()
        var env2 = MountainCar()
        
        let result1 = env1.reset(seed: 42)
        let result2 = env2.reset(seed: 42)
        
        let pos1 = result1.obs[0].item(Float.self)
        let pos2 = result2.obs[0].item(Float.self)
        
        #expect(abs(pos1 - pos2) < 0.0001, "Same seed should produce same initial position")
    }
    
    @Test
    func testResetDifferentSeeds() async throws {
        var env1 = MountainCar()
        var env2 = MountainCar()
        
        let result1 = env1.reset(seed: 42)
        let result2 = env2.reset(seed: 123)
        
        let pos1 = result1.obs[0].item(Float.self)
        let pos2 = result2.obs[0].item(Float.self)
        
        // Different seeds should (very likely) produce different positions
        // Note: There's a tiny chance they could be the same
        #expect(pos1 != pos2, "Different seeds should produce different positions")
    }
    
    @Test
    func testStepShape() async throws {
        var env = MountainCar()
        _ = env.reset()
        
        let result = env.step(1) // No push
        
        #expect(result.obs.shape == [2])
        #expect(result.reward == -1.0)
        #expect(result.truncated == false)
    }
    
    @Test
    func testStepAllActions() async throws {
        var env = MountainCar()
        
        for action in 0..<3 {
            _ = env.reset()
            let result = env.step(action)
            #expect(result.obs.shape == [2], "Action \(action) should return valid observation")
        }
    }
    
    @Test
    func testStepPhysicsPushRight() async throws {
        var env = MountainCar()
        _ = env.reset(seed: 42)
        
        let initialVel = env.state![1].item(Float.self)
        
        // Push right multiple times
        for _ in 0..<10 {
            _ = env.step(2)
        }
        
        let finalVel = env.state![1].item(Float.self)
        
        // After pushing right, velocity should generally increase (become more positive)
        // Note: Gravity effect depends on position
        #expect(finalVel != initialVel, "Velocity should change after pushing")
    }
    
    @Test
    func testStepPhysicsPushLeft() async throws {
        var env = MountainCar()
        _ = env.reset(seed: 42)
        
        let initialVel = env.state![1].item(Float.self)
        
        // Push left multiple times
        for _ in 0..<10 {
            _ = env.step(0)
        }
        
        let finalVel = env.state![1].item(Float.self)
        
        // After pushing left, velocity should generally decrease (become more negative)
        #expect(finalVel != initialVel, "Velocity should change after pushing")
    }
    
    @Test
    func testPositionBounds() async throws {
        var env = MountainCar()
        _ = env.reset()
        
        // Run many steps pushing left to test left boundary
        for _ in 0..<500 {
            let result = env.step(0)
            let position = result.obs[0].item(Float.self)
            
            #expect(position >= env.minPosition, "Position should not go below min")
            #expect(position <= env.maxPosition, "Position should not exceed max")
            
            if result.terminated { break }
        }
    }
    
    @Test
    func testVelocityBounds() async throws {
        var env = MountainCar()
        _ = env.reset()
        
        // Run many steps to check velocity bounds
        for _ in 0..<500 {
            let result = env.step(2) // Push right
            let velocity = result.obs[1].item(Float.self)
            
            #expect(velocity >= -env.maxSpeed, "Velocity should not go below -maxSpeed")
            #expect(velocity <= env.maxSpeed, "Velocity should not exceed maxSpeed")
            
            if result.terminated { break }
        }
    }
    
    @Test
    func testLeftBoundaryStopsVelocity() async throws {
        var env = MountainCar()
        _ = env.reset()
        
        // Push left until we hit the boundary
        var hitBoundary = false
        for _ in 0..<1000 {
            let result = env.step(0)
            let position = result.obs[0].item(Float.self)
            let velocity = result.obs[1].item(Float.self)
            
            if position == env.minPosition {
                // At left boundary, velocity should be >= 0 (stopped or moving right)
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
        var env = MountainCar()
        _ = env.reset()
        
        // Manually set state near goal to test termination
        env.state = MLXArray([env.goalPosition + 0.01, 0.01])
        
        let result = env.step(1) // Any action
        
        #expect(result.terminated == true, "Should terminate when at goal position with non-negative velocity")
    }
    
    @Test
    func testNoTerminationBeforeGoal() async throws {
        var env = MountainCar()
        _ = env.reset()
        
        // Set state just before goal
        env.state = MLXArray([env.goalPosition - 0.1, 0.0])
        
        let result = env.step(1)
        
        #expect(result.terminated == false, "Should not terminate before reaching goal")
    }
    
    @Test
    func testHeightFunctionSinusoidal() async throws {
        // The height function should produce a sinusoidal shape
        // sin(3 * position) * 0.45 + 0.55
        
        let position: Float = 0.0
        let expectedHeight: Float = Foundation.sin(3.0 * position) * 0.45 + 0.55
        let actualHeight = MountainCar.height(at: position)
        
        #expect(abs(actualHeight - expectedHeight) < 0.0001)
    }
    
    @Test
    func testHeightFunctionValley() async throws {
        // The valley (lowest point) should be around position -0.5 (where sin(3x) is minimum)
        // sin(3 * -0.5) = sin(-1.5) ≈ -0.997
        
        let heightAtValley = MountainCar.height(at: -0.5)
        let heightAtLeft = MountainCar.height(at: -1.2)
        let heightAtRight = MountainCar.height(at: 0.6)
        
        #expect(heightAtValley < heightAtLeft, "Valley should be lower than left edge")
        #expect(heightAtValley < heightAtRight, "Valley should be lower than right edge")
    }
    
    @Test
    func testSnapshotCreation() async throws {
        var env = MountainCar()
        _ = env.reset()
        
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
    func testGymnasiumRegistration() async throws {
        Gymnasium.start()
        
        guard var env = Gymnasium.make("MountainCar-v0") as? any Env<MLXArray, Int> else {
            Issue.record("Failed to create MountainCar via Gymnasium.make")
            return
        }
        
        let result = env.reset()
        #expect(result.obs.shape == [2])
    }
    
    @Test @MainActor
    func testGymnasiumMaxEpisodeSteps() async throws {
        Gymnasium.start()
        
        guard let env = Gymnasium.make("MountainCar-v0") as? any Env<MLXArray, Int> else {
            Issue.record("Failed to create MountainCar via Gymnasium.make")
            return
        }
        
        // Default max episode steps should be 200
        #expect(env.spec?.maxEpisodeSteps == 200)
    }
}
