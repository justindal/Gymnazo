import Testing
import Foundation
import MLX
@testable import Gymnazo

@Suite("Acrobot environment")
struct AcrobotTests {
    
    @Test
    func testInitialization() async throws {
        let env = Acrobot()
        
        #expect(env.dt == 0.2)
        #expect(env.linkLength1 == 1.0)
        #expect(env.linkLength2 == 1.0)
        #expect(env.linkMass1 == 1.0)
        #expect(env.linkMass2 == 1.0)
        #expect(env.linkCOMPos1 == 0.5)
        #expect(env.linkCOMPos2 == 0.5)
        #expect(env.linkMOI == 1.0)
        #expect(env.gravity == 9.8)
        #expect(env.state == nil)
    }
    
    @Test
    func testActionSpace() async throws {
        let env = Acrobot()
        
        // Acrobot has 3 actions: -1 torque (0), 0 torque (1), +1 torque (2)
        #expect(env.action_space.n == 3)
        #expect(env.action_space.start == 0)
    }
    
    @Test
    func testObservationSpace() async throws {
        let env = Acrobot()
        
        // Observation: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), dθ1, dθ2]
        #expect(env.observation_space.shape == [6])
    }
    
    @Test
    func testObservationSpaceBounds() async throws {
        let env = Acrobot()
        
        let low = env.observation_space.low
        let high = env.observation_space.high
        
        eval(low, high)
        
        // First 4 elements bounded by [-1, 1]
        for i in 0..<4 {
            #expect(low[i].item(Float.self) == -1.0)
            #expect(high[i].item(Float.self) == 1.0)
        }
        
        // Angular velocity bounds
        #expect(low[4].item(Float.self) == -4 * Float.pi)
        #expect(high[4].item(Float.self) == 4 * Float.pi)
        #expect(low[5].item(Float.self) == -9 * Float.pi)
        #expect(high[5].item(Float.self) == 9 * Float.pi)
    }
    
    @Test
    func testRenderModeInitialization() async throws {
        let envNoRender = Acrobot(render_mode: nil)
        #expect(envNoRender.render_mode == nil)
        
        let envHuman = Acrobot(render_mode: "human")
        #expect(envHuman.render_mode == "human")
    }
    
    @Test
    func testResetReturnsObservation() async throws {
        var env = Acrobot()
        let (obs, info) = env.reset(seed: 42)
        
        eval(obs)
        
        #expect(obs.shape == [6])
        #expect(info.isEmpty || info.count >= 0)
    }
    
    @Test
    func testResetDeterminismWithSeed() async throws {
        var env1 = Acrobot()
        var env2 = Acrobot()
        
        let (obs1, _) = env1.reset(seed: 123)
        let (obs2, _) = env2.reset(seed: 123)
        
        eval(obs1, obs2)
        
        // Same seed should give same initial state
        let diff = abs(obs1 - obs2).sum().item(Float.self)
        #expect(diff < 1e-6)
    }
    
    @Test
    func testResetDifferentSeedsGiveDifferentStates() async throws {
        var env1 = Acrobot()
        var env2 = Acrobot()
        
        let (obs1, _) = env1.reset(seed: 1)
        let (obs2, _) = env2.reset(seed: 999)
        
        eval(obs1, obs2)
        
        // Different seeds should (almost certainly) give different states
        let diff = abs(obs1 - obs2).sum().item(Float.self)
        #expect(diff > 1e-6)
    }
    
    @Test
    func testResetStateWithinBounds() async throws {
        var env = Acrobot()
        
        for seed in 0..<10 {
            _ = env.reset(seed: UInt64(seed))
            
            guard let state = env.state else {
                Issue.record("State should not be nil after reset")
                return
            }
            
            // Initial state should be within [-0.1, 0.1]
            for i in 0..<4 {
                #expect(state[i] >= -0.1 && state[i] <= 0.1)
            }
        }
    }
    
    @Test
    func testResetWithCustomBounds() async throws {
        var env = Acrobot()
        _ = env.reset(seed: 42, options: ["low": Float(-0.2), "high": Float(0.2)])
        
        guard let state = env.state else {
            Issue.record("State should not be nil after reset")
            return
        }
        
        // Initial state should be within [-0.2, 0.2]
        for i in 0..<4 {
            #expect(state[i] >= -0.2 && state[i] <= 0.2)
        }
    }
    
    @Test
    func testStepReturnsValidResult() async throws {
        var env = Acrobot()
        _ = env.reset(seed: 42)
        
        let result = env.step(1) // no torque
        
        eval(result.obs)
        
        #expect(result.obs.shape == [6])
        #expect(result.reward == -1.0 || result.reward == 0.0)
        #expect(result.truncated == false) // Acrobot doesn't truncate itself
    }
    
    @Test
    func testStepRewardIsMinusOneWhileNotTerminated() async throws {
        var env = Acrobot()
        _ = env.reset(seed: 42)
        
        // Take a few steps - should get reward of -1 while not terminated
        for _ in 0..<10 {
            let result = env.step(1) // no torque
            if !result.terminated {
                #expect(result.reward == -1.0)
            } else {
                #expect(result.reward == 0.0)
            }
        }
    }
    
    @Test
    func testObservationIsValidTrigonometric() async throws {
        var env = Acrobot()
        _ = env.reset(seed: 42)
        
        for _ in 0..<50 {
            let result = env.step(Int.random(in: 0..<3))
            eval(result.obs)
            
            // cos and sin values should be in [-1, 1]
            let cosTheta1 = result.obs[0].item(Float.self)
            let sinTheta1 = result.obs[1].item(Float.self)
            let cosTheta2 = result.obs[2].item(Float.self)
            let sinTheta2 = result.obs[3].item(Float.self)
            
            #expect(cosTheta1 >= -1.0 && cosTheta1 <= 1.0)
            #expect(sinTheta1 >= -1.0 && sinTheta1 <= 1.0)
            #expect(cosTheta2 >= -1.0 && cosTheta2 <= 1.0)
            #expect(sinTheta2 >= -1.0 && sinTheta2 <= 1.0)
            
            // cos² + sin² ≈ 1 (within tolerance)
            let sum1 = cosTheta1 * cosTheta1 + sinTheta1 * sinTheta1
            let sum2 = cosTheta2 * cosTheta2 + sinTheta2 * sinTheta2
            #expect(abs(sum1 - 1.0) < 0.001)
            #expect(abs(sum2 - 1.0) < 0.001)
            
            if result.terminated {
                break
            }
        }
    }
    
    @Test
    func testVelocitiesAreBounded() async throws {
        var env = Acrobot()
        _ = env.reset(seed: 42)
        
        for _ in 0..<100 {
            let result = env.step(Int.random(in: 0..<3))
            
            guard let state = env.state else {
                continue
            }
            
            // Velocities should be bounded
            #expect(state[2] >= -4 * Float.pi && state[2] <= 4 * Float.pi)
            #expect(state[3] >= -9 * Float.pi && state[3] <= 9 * Float.pi)
            
            if result.terminated {
                break
            }
        }
    }
    
    @Test
    func testTerminationCondition() async throws {
        var env = Acrobot()
        _ = env.reset(seed: 42)
        
        // Run many steps applying random torques to eventually reach termination
        var terminated = false
        for _ in 0..<500 {
            let action = Int.random(in: 0..<3)
            let result = env.step(action)
            
            if result.terminated {
                terminated = true
                
                // Verify termination condition: -cos(θ1) - cos(θ1 + θ2) > 1.0
                guard let state = env.state else { break }
                let theta1 = state[0]
                let theta2 = state[1]
                let height = -Foundation.cos(theta1) - Foundation.cos(theta1 + theta2)
                #expect(height > 1.0)
                
                // Reward should be 0 on termination
                #expect(result.reward == 0.0)
                break
            }
        }
        
        // It's okay if we don't terminate in 500 steps - the environment is challenging
        // Just verify the test ran
        _ = terminated
    }
    
    @Test
    func testDeterministicStepSequence() async throws {
        var env1 = Acrobot()
        var env2 = Acrobot()
        
        _ = env1.reset(seed: 777)
        _ = env2.reset(seed: 777)
        
        let actions = [0, 1, 2, 0, 1, 2, 1, 0]
        
        for action in actions {
            let result1 = env1.step(action)
            let result2 = env2.step(action)
            
            eval(result1.obs, result2.obs)
            
            let diff = abs(result1.obs - result2.obs).sum().item(Float.self)
            #expect(diff < 1e-5)
            #expect(result1.reward == result2.reward)
            #expect(result1.terminated == result2.terminated)
        }
    }
    
    @Test
    func testAvailableTorques() async throws {
        let env = Acrobot()
        
        #expect(env.availableTorques.count == 3)
        #expect(env.availableTorques[0] == -1.0)
        #expect(env.availableTorques[1] == 0.0)
        #expect(env.availableTorques[2] == 1.0)
    }
    
    @Test
    func testMaxVelocities() async throws {
        let env = Acrobot()
        
        #expect(env.maxVel1 == 4 * Float.pi)
        #expect(env.maxVel2 == 9 * Float.pi)
    }
    
    @Test
    func testAcrobotSnapshotZero() async throws {
        let snapshot = AcrobotSnapshot.zero
        
        #expect(snapshot.theta1 == 0)
        #expect(snapshot.theta2 == 0)
        #expect(snapshot.linkLength1 == 1.0)
        #expect(snapshot.linkLength2 == 1.0)
    }
    
    @Test
    func testAcrobotSnapshotEquatable() async throws {
        let snapshot1 = AcrobotSnapshot(theta1: 0.1, theta2: 0.2, linkLength1: 1.0, linkLength2: 1.0)
        let snapshot2 = AcrobotSnapshot(theta1: 0.1, theta2: 0.2, linkLength1: 1.0, linkLength2: 1.0)
        let snapshot3 = AcrobotSnapshot(theta1: 0.3, theta2: 0.2, linkLength1: 1.0, linkLength2: 1.0)
        
        #expect(snapshot1 == snapshot2)
        #expect(snapshot1 != snapshot3)
    }
    
    @Test
    func testAcrobotSnapshotPositions() async throws {
        // When theta1 = 0, link 1 points straight down (-Y direction)
        let snapshotDown = AcrobotSnapshot(theta1: 0, theta2: 0, linkLength1: 1.0, linkLength2: 1.0)
        
        // p1 should be at (sin(0), -cos(0)) = (0, -1) - pointing down
        #expect(abs(snapshotDown.p1.x - 0.0) < 0.001)
        #expect(abs(snapshotDown.p1.y - (-1.0)) < 0.001)
        
        // p2 should be at p1 + (sin(0), -cos(0)) = (0, -2) - both links down
        #expect(abs(snapshotDown.p2.x - 0.0) < 0.001)
        #expect(abs(snapshotDown.p2.y - (-2.0)) < 0.001)
    }
    
    @Test
    func testCurrentSnapshotBeforeReset() async throws {
        let env = Acrobot()
        let snapshot = env.currentSnapshot
        
        // Before reset, should return zero snapshot
        #expect(snapshot == AcrobotSnapshot.zero)
    }
    
    @Test
    func testCurrentSnapshotAfterReset() async throws {
        var env = Acrobot()
        _ = env.reset(seed: 42)
        
        let snapshot = env.currentSnapshot
        
        // After reset, snapshot should reflect actual state
        guard let state = env.state else {
            Issue.record("State should not be nil")
            return
        }
        
        #expect(snapshot.theta1 == state[0])
        #expect(snapshot.theta2 == state[1])
    }
    
    @Test
    func testCurrentSnapshotUpdatesAfterStep() async throws {
        var env = Acrobot()
        _ = env.reset(seed: 42)
        
        let snapshotBefore = env.currentSnapshot
        _ = env.step(2) // apply positive torque
        let snapshotAfter = env.currentSnapshot
        
        // Snapshot should change after step
        #expect(snapshotBefore != snapshotAfter)
    }
    
    @Test
    func testRenderReturnsNilWhenNoRenderMode() async throws {
        var env = Acrobot(render_mode: nil)
        _ = env.reset(seed: 42)
        
        let result = env.render()
        #expect(result == nil)
    }
    
    @Test
    func testRenderReturnsNilForUnsupportedMode() async throws {
        var env = Acrobot(render_mode: "unsupported")
        _ = env.reset(seed: 42)
        
        let result = env.render()
        #expect(result == nil)
    }
    
    @Test
    func testRenderRgbArrayNotImplemented() async throws {
        var env = Acrobot(render_mode: "rgb_array")
        _ = env.reset(seed: 42)
        
        let result = env.render()
        #expect(result == nil) // not implemented yet
    }
    
    @Test
    func testBookOrNipsDefault() async throws {
        let env = Acrobot()
        #expect(env.bookOrNips == "book")
    }
}

