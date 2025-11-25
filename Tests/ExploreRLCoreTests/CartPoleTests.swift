import Testing
import MLX
@testable import ExploreRLCore

@Suite("CartPole environment")
struct CartPoleTests {
    
    @Test
    func testInitialization() async throws {
        let env = CartPole()
        
        #expect(env.gravity == 9.8)
        #expect(env.masscart == 1.0)
        #expect(env.masspole == 0.1)
        #expect(env.force_mag == 10.0)
        #expect(env.tau == 0.02)
        #expect(env.state == nil)
    }
    
    @Test
    func testActionSpace() async throws {
        let env = CartPole()
        
        // CartPole has 2 actions: push left (0) or push right (1)
        #expect(env.action_space.n == 2)
        #expect(env.action_space.start == 0)
    }
    
    @Test
    func testObservationSpace() async throws {
        let env = CartPole()
        
        // Observation: [x, x_dot, theta, theta_dot]
        #expect(env.observation_space.shape == [4])
    }
    
    @Test
    func testRenderModeInitialization() async throws {
        let envNoRender = CartPole(render_mode: nil)
        #expect(envNoRender.render_mode == nil)
        
        let envHuman = CartPole(render_mode: "human")
        #expect(envHuman.render_mode == "human")
    }
    
    @Test
    func testResetReturnsObservation() async throws {
        var env = CartPole()
        let (obs, info) = env.reset(seed: 42)
        
        #expect(obs.shape == [4])
        #expect(info.isEmpty || info.count >= 0) // info can be empty
    }
    
    @Test
    func testResetDeterminismWithSeed() async throws {
        var env1 = CartPole()
        var env2 = CartPole()
        
        let (obs1, _) = env1.reset(seed: 123)
        let (obs2, _) = env2.reset(seed: 123)
        
        eval(obs1, obs2)
        
        // Same seed should give same initial state
        let diff = abs(obs1 - obs2).sum().item(Float.self)
        #expect(diff < 1e-6)
    }
    
    @Test
    func testResetDifferentSeedsGiveDifferentStates() async throws {
        var env1 = CartPole()
        var env2 = CartPole()
        
        let (obs1, _) = env1.reset(seed: 1)
        let (obs2, _) = env2.reset(seed: 999)
        
        eval(obs1, obs2)
        
        // Different seeds should (almost certainly) give different states
        let diff = abs(obs1 - obs2).sum().item(Float.self)
        #expect(diff > 1e-6)
    }
    
    @Test
    func testResetStateWithinBounds() async throws {
        var env = CartPole()
        
        for seed in 0..<10 {
            let (obs, _) = env.reset(seed: UInt64(seed))
            eval(obs)
            
            // Initial state should be within [-0.05, 0.05]
            let x = obs[0].item(Float.self)
            let xDot = obs[1].item(Float.self)
            let theta = obs[2].item(Float.self)
            let thetaDot = obs[3].item(Float.self)
            
            #expect(x >= -0.05 && x <= 0.05)
            #expect(xDot >= -0.05 && xDot <= 0.05)
            #expect(theta >= -0.05 && theta <= 0.05)
            #expect(thetaDot >= -0.05 && thetaDot <= 0.05)
        }
    }
    
    @Test
    func testStepReturnsValidResult() async throws {
        var env = CartPole()
        _ = env.reset(seed: 42)
        
        let result = env.step(1) // push right
        
        #expect(result.obs.shape == [4])
        #expect(result.reward >= 0)
        #expect(result.truncated == false) // CartPole doesn't truncate
    }
    
    @Test
    func testStepPushLeft() async throws {
        var env = CartPole()
        _ = env.reset(seed: 0)
        
        // Get initial x position
        let initialX = env.state![0].item(Float.self)
        
        // Push left multiple times
        for _ in 0..<5 {
            _ = env.step(0) // push left
        }
        
        let finalX = env.state![0].item(Float.self)
        
        // Cart should have moved left (negative x direction)
        #expect(finalX < initialX)
    }
    
    @Test
    func testStepPushRight() async throws {
        var env = CartPole()
        _ = env.reset(seed: 0)
        
        let initialX = env.state![0].item(Float.self)
        
        // Push right multiple times
        for _ in 0..<5 {
            _ = env.step(1) // push right
        }
        
        let finalX = env.state![0].item(Float.self)
        
        // Cart should have moved right (positive x direction)
        #expect(finalX > initialX)
    }
    
    @Test
    func testStepRewardIsOneWhileBalancing() async throws {
        var env = CartPole()
        _ = env.reset(seed: 42)
        
        // Take a few steps - should get reward of 1 while still balancing
        for _ in 0..<10 {
            let result = env.step(1)
            if !result.terminated {
                #expect(result.reward == 1.0)
            }
        }
    }
    
    @Test
    func testTerminationWhenPoleAngleExceeded() async throws {
        var env = CartPole()
        _ = env.reset(seed: 42)
        
        // Keep pushing in one direction to eventually fail
        var terminated = false
        for _ in 0..<500 {
            let result = env.step(0) // always push left
            if result.terminated {
                terminated = true
                break
            }
        }
        
        #expect(terminated == true)
    }
    
    @Test
    func testTerminationWhenCartPositionExceeded() async throws {
        var env = CartPole()
        _ = env.reset(seed: 0)
        
        // This might terminate from position or angle
        var terminated = false
        for _ in 0..<1000 {
            let result = env.step(1) // always push right
            if result.terminated {
                terminated = true
                break
            }
        }
        
        #expect(terminated == true)
    }
    
    @Test
    func testCartPoleSnapshotZero() async throws {
        let snapshot = CartPoleSnapshot.zero
        
        #expect(snapshot.x == 0)
        #expect(snapshot.theta == 0)
        #expect(snapshot.x_threshold == 2.4)
    }
    
    @Test
    func testCartPoleSnapshotEquatable() async throws {
        let snapshot1 = CartPoleSnapshot(x: 1.0, theta: 0.1, x_threshold: 2.4)
        let snapshot2 = CartPoleSnapshot(x: 1.0, theta: 0.1, x_threshold: 2.4)
        let snapshot3 = CartPoleSnapshot(x: 2.0, theta: 0.1, x_threshold: 2.4)
        
        #expect(snapshot1 == snapshot2)
        #expect(snapshot1 != snapshot3)
    }
    
    @Test
    func testCurrentSnapshotBeforeReset() async throws {
        let env = CartPole()
        let snapshot = env.currentSnapshot
        
        // Before reset, should return zero snapshot
        #expect(snapshot == CartPoleSnapshot.zero)
    }
    
    @Test
    func testCurrentSnapshotAfterReset() async throws {
        var env = CartPole()
        _ = env.reset(seed: 42)
        
        let snapshot = env.currentSnapshot
        
        // After reset, snapshot should reflect actual state
        let expectedX = env.state![0].item(Float.self)
        let expectedTheta = env.state![2].item(Float.self)
        
        #expect(snapshot.x == expectedX)
        #expect(snapshot.theta == expectedTheta)
        #expect(snapshot.x_threshold == env.x_threshold)
    }
    
    @Test
    func testCurrentSnapshotUpdatesAfterStep() async throws {
        var env = CartPole()
        _ = env.reset(seed: 42)
        
        let snapshotBefore = env.currentSnapshot
        _ = env.step(1)
        let snapshotAfter = env.currentSnapshot
        
        // Snapshot should change after step
        #expect(snapshotBefore != snapshotAfter)
    }
    
    @Test
    func testRenderReturnsNilWhenNoRenderMode() async throws {
        var env = CartPole(render_mode: nil)
        _ = env.reset(seed: 42)
        
        let result = env.render()
        #expect(result == nil)
    }
    
    @Test
    func testRenderReturnsNilForUnsupportedMode() async throws {
        var env = CartPole(render_mode: "unsupported")
        _ = env.reset(seed: 42)
        
        let result = env.render()
        #expect(result == nil)
    }
    
    @Test
    func testRenderRgbArrayNotImplemented() async throws {
        var env = CartPole(render_mode: "rgb_array")
        _ = env.reset(seed: 42)
        
        let result = env.render()
        #expect(result == nil) // not implemented yet
    }
    
    @Test
    func testPhysicsConstants() async throws {
        let env = CartPole()
        
        #expect(env.total_mass == env.masscart + env.masspole)
        #expect(env.polemass_length == env.masspole * env.length)
    }
    
    @Test
    func testThresholds() async throws {
        let env = CartPole()
        
        #expect(env.x_threshold == 2.4)
        // theta_threshold_radians ≈ 12 degrees in radians
        let expectedThetaThreshold = 12.0 * 2.0 * Float.pi / 360.0
        #expect(abs(env.theta_threshold_radians - expectedThetaThreshold) < 0.001)
    }
    
    @Test
    func testDeterministicStepSequence() async throws {
        var env1 = CartPole()
        var env2 = CartPole()
        
        _ = env1.reset(seed: 777)
        _ = env2.reset(seed: 777)
        
        let actions = [1, 0, 1, 1, 0, 0, 1, 0]
        
        for action in actions {
            let result1 = env1.step(action)
            let result2 = env2.step(action)
            
            eval(result1.obs, result2.obs)
            
            let diff = abs(result1.obs - result2.obs).sum().item(Float.self)
            #expect(diff < 1e-6)
            #expect(result1.reward == result2.reward)
            #expect(result1.terminated == result2.terminated)
        }
    }
}

