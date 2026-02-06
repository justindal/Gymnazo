import Testing
import MLX
@testable import Gymnazo

@Suite("CartPole environment")
struct CartPoleTests {
    func makeCartPole(renderMode: RenderMode? = nil) async throws -> CartPole {
        let options: EnvOptions = renderMode.map { ["render_mode": $0.rawValue] } ?? [:]
        let env = try await Gymnazo.make("CartPole", options: options)
        guard let cartPole = env.unwrapped as? CartPole else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "CartPole",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return cartPole
    }
    
    @Test
    func testInitialization() async throws {
        let env = try await makeCartPole()
        
        #expect(env.gravity == 9.8)
        #expect(env.masscart == 1.0)
        #expect(env.masspole == 0.1)
        #expect(env.force_mag == 10.0)
        #expect(env.tau == 0.02)
        #expect(env.state == nil)
    }
    
    @Test
    func testActionSpace() async throws {
        let env = try await makeCartPole()

        guard let actionSpace = env.actionSpace as? Discrete else {
            Issue.record("Action space is not Discrete")
            return
        }

        #expect(actionSpace.n == 2)
        #expect(actionSpace.start == 0)
    }
    
    @Test
    func testObservationSpace() async throws {
        let env = try await makeCartPole()
        
        #expect(env.observationSpace.shape == [4])
    }
    
    @Test
    func testRenderModeInitialization() async throws {
        let envNoRender = try await makeCartPole(renderMode: nil)
        #expect(envNoRender.renderMode == nil)
        
        let envHuman = try await makeCartPole(renderMode: .human)
        #expect(envHuman.renderMode == .human)
    }
    
    @Test
    func testResetReturnsObservation() async throws {
        var env = try await makeCartPole()
        let result = try env.reset(seed: 42)
        let obs = result.obs
        let info = result.info
        
        #expect(obs.shape == [4])
        #expect(info.isEmpty || info.count >= 0)
    }
    
    @Test
    func testResetDeterminismWithSeed() async throws {
        var env1 = try await makeCartPole()
        var env2 = try await makeCartPole()
        
        let obs1 = try env1.reset(seed: 123).obs
        let obs2 = try env2.reset(seed: 123).obs
        
        eval(obs1, obs2)
        
        let diff = abs(obs1 - obs2).sum().item(Float.self)
        #expect(diff < 1e-6)
    }
    
    @Test
    func testResetDifferentSeedsGiveDifferentStates() async throws {
        var env1 = try await makeCartPole()
        var env2 = try await makeCartPole()
        
        let obs1 = try env1.reset(seed: 1).obs
        let obs2 = try env2.reset(seed: 999).obs
        
        eval(obs1, obs2)
        
        let diff = abs(obs1 - obs2).sum().item(Float.self)
        #expect(diff > 1e-6)
    }
    
    @Test
    func testResetStateWithinBounds() async throws {
        var env = try await makeCartPole()
        
        for seed in 0..<10 {
            let obs = try env.reset(seed: UInt64(seed)).obs
            eval(obs)
            
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
        var env = try await makeCartPole()
        _ = try env.reset(seed: 42)
        
        let result = try env.step(MLXArray(Int32(1)))
        
        #expect(result.obs.shape == [4])
        #expect(result.reward >= 0)
        #expect(result.truncated == false)
    }
    
    @Test
    func testStepPushLeft() async throws {
        var env = try await makeCartPole()
        _ = try env.reset(seed: 0)
        
        let initialX = env.state![0].item(Float.self)
        
        for _ in 0..<5 {
            _ = try env.step(MLXArray(Int32(0)))
        }
        
        let finalX = env.state![0].item(Float.self)
        
        #expect(finalX < initialX)
    }
    
    @Test
    func testStepPushRight() async throws {
        var env = try await makeCartPole()
        _ = try env.reset(seed: 0)
        
        let initialX = env.state![0].item(Float.self)
        
        for _ in 0..<5 {
            _ = try env.step(MLXArray(Int32(1)))
        }
        
        let finalX = env.state![0].item(Float.self)
        
        #expect(finalX > initialX)
    }
    
    @Test
    func testStepRewardIsOneWhileBalancing() async throws {
        var env = try await makeCartPole()
        _ = try env.reset(seed: 42)
        
        for _ in 0..<10 {
            let result = try env.step(MLXArray(Int32(1)))
            if !result.terminated {
                #expect(result.reward == 1.0)
            }
        }
    }
    
    @Test
    func testTerminationWhenPoleAngleExceeded() async throws {
        var env = try await makeCartPole()
        _ = try env.reset(seed: 42)
        
        var terminated = false
        for _ in 0..<500 {
            let result = try env.step(MLXArray(Int32(0)))
            if result.terminated {
                terminated = true
                break
            }
        }
        
        #expect(terminated == true)
    }
    
    @Test
    func testTerminationWhenCartPositionExceeded() async throws {
        var env = try await makeCartPole()
        _ = try env.reset(seed: 0)
        
        var terminated = false
        for _ in 0..<1000 {
            let result = try env.step(MLXArray(Int32(1)))
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
        let env = try await makeCartPole()
        let snapshot = env.currentSnapshot
        
        #expect(snapshot == CartPoleSnapshot.zero)
    }
    
    @Test
    func testCurrentSnapshotAfterReset() async throws {
        var env = try await makeCartPole()
        _ = try env.reset(seed: 42)
        
        let snapshot = env.currentSnapshot
        
        let expectedX = env.state![0].item(Float.self)
        let expectedTheta = env.state![2].item(Float.self)
        
        #expect(snapshot.x == expectedX)
        #expect(snapshot.theta == expectedTheta)
        #expect(snapshot.x_threshold == env.x_threshold)
    }
    
    @Test
    func testCurrentSnapshotUpdatesAfterStep() async throws {
        var env = try await makeCartPole()
        _ = try env.reset(seed: 42)
        
        let snapshotBefore = env.currentSnapshot
        _ = try env.step(MLXArray(Int32(1)))
        let snapshotAfter = env.currentSnapshot
        
        #expect(snapshotBefore != snapshotAfter)
    }
    
    @Test
    func testRenderReturnsNilWhenNoRenderMode() async throws {
        var env = try await makeCartPole(renderMode: nil)
        _ = try env.reset(seed: 42)
        
        let result = try env.render()
        #expect(result == nil)
    }
    
    @Test
    func testRenderReturnsNilForAnsiMode() async throws {
        var env = try await makeCartPole(renderMode: .ansi)
        _ = try env.reset(seed: 42)
        
        let result = try env.render()
        #expect(result == nil)
    }
    
    @Test
    func testRenderRgbArrayNotImplemented() async throws {
        var env = try await makeCartPole(renderMode: .rgbArray)
        _ = try env.reset(seed: 42)
        
        let result = try env.render()
        #expect(result == nil)
    }
    
    @Test
    func testPhysicsConstants() async throws {
        let env = try await makeCartPole()
        
        #expect(env.total_mass == env.masscart + env.masspole)
        #expect(env.polemass_length == env.masspole * env.length)
    }
    
    @Test
    func testThresholds() async throws {
        let env = try await makeCartPole()
        
        #expect(env.x_threshold == 2.4)
        let expectedThetaThreshold = 12.0 * 2.0 * Float.pi / 360.0
        #expect(abs(env.theta_threshold_radians - expectedThetaThreshold) < 0.001)
    }
    
    @Test
    func testDeterministicStepSequence() async throws {
        var env1 = try await makeCartPole()
        var env2 = try await makeCartPole()
        
        _ = try env1.reset(seed: 777)
        _ = try env2.reset(seed: 777)
        
        let actions = [1, 0, 1, 1, 0, 0, 1, 0]
        
        for action in actions {
            let result1 = try env1.step(MLXArray(Int32(action)))
            let result2 = try env2.step(MLXArray(Int32(action)))
            
            eval(result1.obs, result2.obs)
            
            let diff = abs(result1.obs - result2.obs).sum().item(Float.self)
            #expect(diff < 1e-6)
            #expect(result1.reward == result2.reward)
            #expect(result1.terminated == result2.terminated)
        }
    }
}

