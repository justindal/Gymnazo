//
//  PendulumTests.swift
//

import Testing
import MLX
@testable import Gymnazo

@Suite("Pendulum environment")
struct PendulumTests {
    
    @Test
    func testInitialization() async throws {
        let env = Pendulum()
        
        #expect(env.g == 10.0)
        #expect(env.maxSpeed == 8.0)
        #expect(env.maxTorque == 2.0)
        #expect(env.dt == 0.05)
        #expect(env.m == 1.0)
        #expect(env.l == 1.0)
        #expect(env.state == nil)
    }
    
    @Test
    func testCustomGravity() async throws {
        let env = Pendulum(g: 9.81)
        #expect(env.g == 9.81)
    }
    
    @Test
    func testObservationSpace() async throws {
        let env = Pendulum()
        
        #expect(env.observation_space.shape == [3])
        
        let low = env.observation_space.low.asArray(Float.self)
        let high = env.observation_space.high.asArray(Float.self)
        
        #expect(low[0] == -1.0)
        #expect(low[1] == -1.0)
        #expect(low[2] == -8.0)
        #expect(high[0] == 1.0)
        #expect(high[1] == 1.0)
        #expect(high[2] == 8.0)
    }
    
    @Test
    func testActionSpace() async throws {
        let env = Pendulum()
        
        #expect(env.action_space.shape == [1])
        
        let low = env.action_space.low.asArray(Float.self)
        let high = env.action_space.high.asArray(Float.self)
        
        #expect(low[0] == -2.0)
        #expect(high[0] == 2.0)
    }
    
    @Test
    func testResetReturnsObservation() async throws {
        var env = Pendulum()
        let result = env.reset(seed: 42)
        
        #expect(result.obs.shape == [3])
        #expect(env.state != nil)
        
        let obs = result.obs.asArray(Float.self)
        let cosTheta = obs[0]
        let sinTheta = obs[1]
        let thetaDot = obs[2]
        
        #expect(cosTheta >= -1.0 && cosTheta <= 1.0)
        #expect(sinTheta >= -1.0 && sinTheta <= 1.0)
        #expect(thetaDot >= -1.0 && thetaDot <= 1.0)
        
        let cosSinSumSquared = cosTheta * cosTheta + sinTheta * sinTheta
        #expect(abs(cosSinSumSquared - 1.0) < 0.001)
    }
    
    @Test
    func testResetDeterminismWithSeed() async throws {
        var env1 = Pendulum()
        var env2 = Pendulum()
        
        let result1 = env1.reset(seed: 123)
        let result2 = env2.reset(seed: 123)
        
        let obs1 = result1.obs.asArray(Float.self)
        let obs2 = result2.obs.asArray(Float.self)
        
        #expect(obs1[0] == obs2[0])
        #expect(obs1[1] == obs2[1])
        #expect(obs1[2] == obs2[2])
    }
    
    @Test
    func testResetDifferentSeeds() async throws {
        var env1 = Pendulum()
        var env2 = Pendulum()
        
        let result1 = env1.reset(seed: 111)
        let result2 = env2.reset(seed: 222)
        
        let obs1 = result1.obs.asArray(Float.self)
        let obs2 = result2.obs.asArray(Float.self)
        
        let allEqual = obs1[0] == obs2[0] && obs1[1] == obs2[1] && obs1[2] == obs2[2]
        #expect(!allEqual)
    }
    
    @Test
    func testStepReturnsValidResult() async throws {
        var env = Pendulum()
        _ = env.reset(seed: 42)
        
        let action = MLXArray([0.5] as [Float32])
        let result = env.step(action)
        
        #expect(result.obs.shape == [3])
        #expect(result.terminated == false)
        #expect(result.truncated == false)
    }
    
    @Test
    func testStepRewardRange() async throws {
        var env = Pendulum()
        _ = env.reset(seed: 42)
        
        let action = MLXArray([1.0] as [Float32])
        let result = env.step(action)
        
        #expect(result.reward <= 0.0)
        #expect(result.reward >= -17.0)
    }
    
    @Test
    func testActionClipping() async throws {
        var env = Pendulum()
        _ = env.reset(seed: 42)
        
        let largeAction = MLXArray([10.0] as [Float32])
        let result1 = env.step(largeAction)
        
        var env2 = Pendulum()
        _ = env2.reset(seed: 42)
        let clippedAction = MLXArray([2.0] as [Float32])
        let result2 = env2.step(clippedAction)
        
        let obs1 = result1.obs.asArray(Float.self)
        let obs2 = result2.obs.asArray(Float.self)
        
        #expect(abs(obs1[0] - obs2[0]) < 0.001)
        #expect(abs(obs1[1] - obs2[1]) < 0.001)
        #expect(abs(obs1[2] - obs2[2]) < 0.001)
    }
    
    @Test
    func testAngularVelocityBounded() async throws {
        var env = Pendulum()
        _ = env.reset(seed: 42)
        
        let maxAction = MLXArray([2.0] as [Float32])
        for _ in 0..<1000 {
            let result = env.step(maxAction)
            let thetaDot = result.obs[2].item(Float.self)
            #expect(thetaDot >= -8.0 && thetaDot <= 8.0)
        }
    }
    
    @Test
    func testNeverTerminates() async throws {
        var env = Pendulum()
        _ = env.reset(seed: 42)
        
        for _ in 0..<500 {
            let action = MLXArray([Float.random(in: -2...2)] as [Float32])
            let result = env.step(action)
            #expect(result.terminated == false)
        }
    }
    
    @Test
    func testSnapshotCreation() async throws {
        var env = Pendulum()
        
        #expect(env.currentSnapshot == nil)
        
        _ = env.reset(seed: 42)
        let snapshot = env.currentSnapshot
        
        #expect(snapshot != nil)
        #expect(snapshot == env.currentSnapshot)
    }
    
    @Test
    func testSnapshotEquatable() async throws {
        let s1 = PendulumSnapshot(theta: 0.5, thetaDot: 0.1, torque: nil)
        let s2 = PendulumSnapshot(theta: 0.5, thetaDot: 0.1, torque: nil)
        let s3 = PendulumSnapshot(theta: 0.6, thetaDot: 0.1, torque: nil)
        
        #expect(s1 == s2)
        #expect(s1 != s3)
    }
    
    @Test
    func testSnapshotZero() async throws {
        let zero = PendulumSnapshot.zero
        #expect(zero.theta == 0)
        #expect(zero.thetaDot == 0)
        #expect(zero.torque == nil)
    }
    
    @Test
    func testPhysicsDynamics() async throws {
        var env = Pendulum()
        _ = env.reset(seed: 42, options: ["x_init": Float(0.01), "y_init": Float(0.0)])
        
        let action = MLXArray([0.0] as [Float32])
        _ = env.step(action)
        
        let state = env.state!
        #expect(state.theta != 0.01 || state.thetaDot != 0.0)
    }
    
    @Test
    func testResetWithCustomBounds() async throws {
        var env = Pendulum()
        _ = env.reset(seed: 42, options: ["x_init": Float(0.1), "y_init": Float(0.05)])
        
        let state = env.state!
        #expect(abs(state.theta) <= 0.1)
        #expect(abs(state.thetaDot) <= 0.05)
    }
    
    @Test
    func testRenderReturnsNilWhenNoRenderMode() async throws {
        var env = Pendulum()
        _ = env.reset(seed: 42)
        
        let rendered = env.render()
        #expect(rendered == nil)
    }
    
    @Test
    @MainActor
    func testGymnazoRegistration() async throws {
        guard var env = Gymnazo.make("Pendulum") as? any Env<MLXArray, MLXArray> else {
            Issue.record("Failed to create Pendulum via Gymnazo.make")
            return
        }
        
        let result = env.reset()
        #expect(result.obs.shape == [3])
    }
    
    @Test
    @MainActor
    func testGymnazoMaxEpisodeSteps() async throws {
        guard let env = Gymnazo.make("Pendulum") as? any Env<MLXArray, MLXArray> else {
            Issue.record("Failed to create Pendulum via Gymnazo.make")
            return
        }
        
        #expect(env.spec?.maxEpisodeSteps == 200)
    }
    
    @Test
    @MainActor
    func testGymnazoCustomGravity() async throws {
        guard let env = Gymnazo.make("Pendulum", kwargs: ["g": 9.81]) as? any Env<MLXArray, MLXArray> else {
            Issue.record("Failed to create Pendulum via Gymnazo.make")
            return
        }
        
        if let pendulum = env.unwrapped as? Pendulum {
            #expect(pendulum.g == 9.81)
        } else {
            Issue.record("Could not unwrap to Pendulum")
        }
    }
}
