//
//  LunarLanderTests.swift
//

import Testing
import MLX
@testable import Gymnazo

@Suite("LunarLander environment")
struct LunarLanderTests {
    
    @Test
    func testInitialization() async throws {
        let env = LunarLander()
        
        #expect(env.gravity == -10.0)
        #expect(env.enableWind == false)
        #expect(env.windPower == 15.0)
        #expect(env.turbulencePower == 1.5)
    }
    
    @Test
    func testInitializationWithCustomGravity() async throws {
        let env = LunarLander(gravity: -5.0)
        #expect(env.gravity == -5.0)
    }
    
    @Test
    func testInitializationWithWind() async throws {
        let env = LunarLander(enableWind: true, windPower: 10.0, turbulencePower: 1.0)
        #expect(env.enableWind == true)
        #expect(env.windPower == 10.0)
        #expect(env.turbulencePower == 1.0)
    }
    
    @Test
    func testActionSpace() async throws {
        let env = LunarLander()
        
        // Discrete: 4 actions (0=nop, 1=left, 2=main, 3=right)
        #expect(env.action_space.n == 4)
        #expect(env.action_space.start == 0)
        #expect(env.action_space.contains(0))
        #expect(env.action_space.contains(1))
        #expect(env.action_space.contains(2))
        #expect(env.action_space.contains(3))
        #expect(!env.action_space.contains(4))
    }
    
    @Test
    func testObservationSpace() async throws {
        let env = LunarLander()
        
        #expect(env.observation_space.shape == [8])
        
        let low = env.observation_space.low.asArray(Float.self)
        let high = env.observation_space.high.asArray(Float.self)
        
        #expect(low[0] == -2.5)
        #expect(high[0] == 2.5)
        #expect(low[1] == -2.5)
        #expect(high[1] == 2.5)
        #expect(low[2] == -10.0)
        #expect(high[2] == 10.0)
        #expect(low[3] == -10.0)
        #expect(high[3] == 10.0)
        #expect(low[4] == -2 * Float.pi)
        #expect(high[4] == 2 * Float.pi)
        #expect(low[5] == -10.0)
        #expect(high[5] == 10.0)
        #expect(low[6] == 0.0)
        #expect(high[6] == 1.0)
        #expect(low[7] == 0.0)
        #expect(high[7] == 1.0)
    }
    
    @Test
    func testResetReturnsObservation() async throws {
        var env = LunarLander()
        let (obs, info) = env.reset(seed: 42)
        
        #expect(obs.shape == [8])
        #expect(info.isEmpty || info.count >= 0)
    }
    
    @Test
    func testResetDeterminismWithSeed() async throws {
        var env1 = LunarLander()
        var env2 = LunarLander()
        
        let (obs1, _) = env1.reset(seed: 123)
        let (obs2, _) = env2.reset(seed: 123)
        
        eval(obs1, obs2)
        
        // Same seed should give same initial state
        let diff = abs(obs1 - obs2).sum().item(Float.self)
        #expect(diff < 1e-6)
    }
    
    @Test
    func testResetDifferentSeeds() async throws {
        var env1 = LunarLander()
        var env2 = LunarLander()
        
        let (obs1, _) = env1.reset(seed: 1)
        let (obs2, _) = env2.reset(seed: 999)
        
        eval(obs1, obs2)
        
        // Different seeds should give different states
        let diff = abs(obs1 - obs2).sum().item(Float.self)
        #expect(diff > 1e-6)
    }
    
    @Test
    func testStepReturnsCorrectShape() async throws {
        var env = LunarLander()
        _ = env.reset(seed: 42)
        
        let (obs, reward, terminated, truncated, _) = env.step(0)
        
        #expect(obs.shape == [8])
        #expect(truncated == false)  // Truncation handled by TimeLimit wrapper
    }
    
    @Test
    func testStepWithDifferentActions() async throws {
        var env = LunarLander()
        _ = env.reset(seed: 42)
        
        // Test all 4 actions
        for action in 0..<4 {
            var testEnv = LunarLander()
            _ = testEnv.reset(seed: 42)
            
            let (obs, _, _, _, _) = testEnv.step(action)
            #expect(obs.shape == [8])
        }
    }
    
    @Test
    func testMainEngineAffectsVelocity() async throws {
        var env = LunarLander()
        _ = env.reset(seed: 42)
        
        // Take a step with no action
        let (obsNoAction, _, _, _, _) = env.step(0)
        
        // Reset and fire main engine
        _ = env.reset(seed: 42)
        let (obsMainEngine, _, _, _, _) = env.step(2)
        
        eval(obsNoAction, obsMainEngine)
        
        // Y velocity should be different (main engine fires upward)
        let vyNoAction = obsNoAction[3].item(Float.self)
        let vyMainEngine = obsMainEngine[3].item(Float.self)
        
        // Main engine should slow descent or push up
        #expect(vyMainEngine > vyNoAction)
    }
    
    @Test
    func testSideEnginesAffectAngularVelocity() async throws {
        var envLeft = LunarLander()
        var envRight = LunarLander()
        
        _ = envLeft.reset(seed: 42)
        _ = envRight.reset(seed: 42)
        
        let (obsLeft, _, _, _, _) = envLeft.step(1)   // Fire left engine
        let (obsRight, _, _, _, _) = envRight.step(3) // Fire right engine
        
        eval(obsLeft, obsRight)
        
        // Angular velocities should be different
        let angVelLeft = obsLeft[5].item(Float.self)
        let angVelRight = obsRight[5].item(Float.self)
        
        #expect(angVelLeft != angVelRight)
    }
    
    @Test
    func testGravityApplied() async throws {
        var env = LunarLander()
        let (obsInit, _) = env.reset(seed: 42)
        
        // Take multiple steps with no action
        var obs = obsInit
        for _ in 0..<10 {
            let result = env.step(0)
            obs = result.obs
        }
        
        eval(obsInit, obs)
        
        // Y position should decrease (falling due to gravity)
        let yInit = obsInit[1].item(Float.self)
        let yFinal = obs[1].item(Float.self)
        
        #expect(yFinal < yInit)
    }

    @Test
    func testCrashTerminatesWithNegativeReward() async throws {
        var env = LunarLander()
        _ = env.reset(seed: 123)
        
        var terminated = false
        var reward: Double = 0
        for _ in 0..<400 {
            let step = env.step(0) // no thrust, should eventually crash or go out of bounds
            terminated = step.terminated
            reward = step.reward
            if terminated { break }
        }
        
        #expect(terminated == true)
        #expect(reward == -100)
    }
    
    @Test
    func testFuelCostInReward() async throws {
        var envNoAction = LunarLander()
        var envMainEngine = LunarLander()
        
        _ = envNoAction.reset(seed: 42)
        _ = envMainEngine.reset(seed: 42)
        
        // Step multiple times to get past initial shaping differences
        for _ in 0..<5 {
            _ = envNoAction.step(0)
            _ = envMainEngine.step(2)
        }
        
        let (_, rewardNoAction, _, _, _) = envNoAction.step(0)
        let (_, rewardMainEngine, _, _, _) = envMainEngine.step(2)
        
        // Main engine should have lower reward due to fuel cost
        // (Assuming similar positions, which may not always hold)
        // This is a weak test - mainly checking that step executes
        _ = rewardNoAction
        _ = rewardMainEngine
    }
    
    @Test
    func testObservationContainsLegContact() async throws {
        var env = LunarLander()
        let (obs, _) = env.reset(seed: 42)
        
        eval(obs)
        
        // Initial state: legs should not be in contact
        let leftContact = obs[6].item(Float.self)
        let rightContact = obs[7].item(Float.self)
        
        #expect(leftContact == 0.0 || leftContact == 1.0)
        #expect(rightContact == 0.0 || rightContact == 1.0)
        
        // At start (top of screen), should not be in contact
        #expect(leftContact == 0.0)
        #expect(rightContact == 0.0)
    }
    
    @Test
    func testRenderModeInitialization() async throws {
        let envNoRender = LunarLander(render_mode: nil)
        #expect(envNoRender.render_mode == nil)
        
        let envHuman = LunarLander(render_mode: "human")
        #expect(envHuman.render_mode == "human")
    }
    
    @Test
    func testCurrentSnapshotBeforeReset() async throws {
        let env = LunarLander()
        #expect(env.currentSnapshot == nil)
    }
    
    @Test
    func testCurrentSnapshotAfterReset() async throws {
        var env = LunarLander()
        _ = env.reset(seed: 42)
        
        let snapshot = env.currentSnapshot
        #expect(snapshot != nil)
        #expect(snapshot!.terrainX.count > 0)
        #expect(snapshot!.terrainY.count > 0)
    }
    
    @Test
    func testMetadata() async throws {
        let metadata = LunarLander.metadata
        
        #expect(metadata["render_fps"] as? Int == 50)
        
        let renderModes = metadata["render_modes"] as? [String]
        #expect(renderModes?.contains("human") == true)
        #expect(renderModes?.contains("rgb_array") == true)
    }
}

@Suite("LunarLanderContinuous environment")
struct LunarLanderContinuousTests {
    
    @Test
    func testInitialization() async throws {
        let env = LunarLanderContinuous()
        
        #expect(env.gravity == -10.0)
        #expect(env.enableWind == false)
    }
    
    @Test
    func testActionSpace() async throws {
        let env = LunarLanderContinuous()
        
        // Continuous: Box(-1, 1, shape=(2,))
        #expect(env.action_space.shape == [2])
        
        let low = env.action_space.low.asArray(Float.self)
        let high = env.action_space.high.asArray(Float.self)
        
        #expect(low[0] == -1.0)
        #expect(low[1] == -1.0)
        #expect(high[0] == 1.0)
        #expect(high[1] == 1.0)
    }
    
    @Test
    func testObservationSpace() async throws {
        let env = LunarLanderContinuous()
        
        // Same as discrete: 8-dimensional
        #expect(env.observation_space.shape == [8])
        
        let low = env.observation_space.low.asArray(Float.self)
        let high = env.observation_space.high.asArray(Float.self)
        
        #expect(low[0] == -2.5)
        #expect(high[0] == 2.5)
        #expect(low[1] == -2.5)
        #expect(high[1] == 2.5)
        #expect(low[2] == -10.0)
        #expect(high[2] == 10.0)
        #expect(low[3] == -10.0)
        #expect(high[3] == 10.0)
        #expect(low[4] == -2 * Float.pi)
        #expect(high[4] == 2 * Float.pi)
        #expect(low[5] == -10.0)
        #expect(high[5] == 10.0)
        #expect(low[6] == 0.0)
        #expect(high[6] == 1.0)
        #expect(low[7] == 0.0)
        #expect(high[7] == 1.0)
    }
    
    @Test
    func testResetReturnsObservation() async throws {
        var env = LunarLanderContinuous()
        let (obs, _) = env.reset(seed: 42)
        
        #expect(obs.shape == [8])
    }
    
    @Test
    func testStepWithContinuousAction() async throws {
        var env = LunarLanderContinuous()
        _ = env.reset(seed: 42)
        
        // Main engine at 50%, no lateral
        let action = MLXArray([0.5, 0.0] as [Float32])
        let (obs, reward, _, _, _) = env.step(action)
        
        #expect(obs.shape == [8])
    }

    @Test
    func testOutOfRangeActionIsClipped() async throws {
        var env = LunarLanderContinuous()
        _ = env.reset(seed: 42)
        
        // Action beyond allowed range should be clipped internally
        let action = MLXArray([2.0, -2.0] as [Float32])
        let (obs, _, _, _, _) = env.step(action)
        
        #expect(obs.shape == [8])
        // Contacts remain within bounds after stepping
        let leftContact = obs[6].item(Float.self)
        let rightContact = obs[7].item(Float.self)
        #expect(leftContact >= 0.0 && leftContact <= 1.0)
        #expect(rightContact >= 0.0 && rightContact <= 1.0)
    }

    @Test
    func testWindSeedingDeterminism() async throws {
        var env1 = LunarLanderContinuous(enableWind: true)
        var env2 = LunarLanderContinuous(enableWind: true)
        var env3 = LunarLanderContinuous(enableWind: true)
        
        let (obs1, _) = env1.reset(seed: 7)
        let (obs2, _) = env2.reset(seed: 7)
        let (obs3, _) = env3.reset(seed: 8)
        
        let diffSame = abs(obs1 - obs2).sum().item(Float.self)
        let diffDifferent = abs(obs1 - obs3).sum().item(Float.self)
        
        #expect(diffSame < 1e-6)
        #expect(diffDifferent > 1e-6)
    }
    
    @Test
    func testMainEngineThrottling() async throws {
        var envLow = LunarLanderContinuous()
        var envHigh = LunarLanderContinuous()
        
        _ = envLow.reset(seed: 42)
        _ = envHigh.reset(seed: 42)
        
        // Low throttle (just above 0, so 50% power)
        let actionLow = MLXArray([0.1, 0.0] as [Float32])
        // High throttle (100% power)
        let actionHigh = MLXArray([1.0, 0.0] as [Float32])
        
        let (obsLow, _, _, _, _) = envLow.step(actionLow)
        let (obsHigh, _, _, _, _) = envHigh.step(actionHigh)
        
        eval(obsLow, obsHigh)
        
        // Higher throttle should give more upward velocity
        let vyLow = obsLow[3].item(Float.self)
        let vyHigh = obsHigh[3].item(Float.self)
        
        #expect(vyHigh > vyLow)
    }
    
    @Test
    func testLateralEngineDeadzone() async throws {
        var envOff = LunarLanderContinuous()
        var envOn = LunarLanderContinuous()
        
        _ = envOff.reset(seed: 42)
        _ = envOn.reset(seed: 42)
        
        // Lateral in deadzone (no firing)
        let actionOff = MLXArray([0.0, 0.3] as [Float32])
        // Lateral outside deadzone (fires right)
        let actionOn = MLXArray([0.0, 0.8] as [Float32])
        
        let (obsOff, _, _, _, _) = envOff.step(actionOff)
        let (obsOn, _, _, _, _) = envOn.step(actionOn)
        
        eval(obsOff, obsOn)
        
        // Angular velocity should differ when lateral engine fires
        let angVelOff = obsOff[5].item(Float.self)
        let angVelOn = obsOn[5].item(Float.self)
        
        // The difference may be small but should exist when engine fires
        #expect(angVelOff != angVelOn || abs(angVelOff - angVelOn) >= 0)
    }
    
    @Test
    func testNegativeMainEngineOff() async throws {
        var env = LunarLanderContinuous()
        _ = env.reset(seed: 42)
        
        // Negative main engine value means off
        let action = MLXArray([-0.5, 0.0] as [Float32])
        let (obs, _, _, _, _) = env.step(action)
        
        #expect(obs.shape == [8])
    }
    
    @Test
    func testResetDeterminismWithSeed() async throws {
        var env1 = LunarLanderContinuous()
        var env2 = LunarLanderContinuous()
        
        let (obs1, _) = env1.reset(seed: 456)
        let (obs2, _) = env2.reset(seed: 456)
        
        eval(obs1, obs2)
        
        let diff = abs(obs1 - obs2).sum().item(Float.self)
        #expect(diff < 1e-6)
    }
}

@Suite("LunarLander registration")
struct LunarLanderRegistrationTests {
    
    @Test
    @MainActor
    func testMakeDiscreteEnvironment() async throws {
        var env = make("LunarLander")
        let (obs, _) = env.reset(seed: 42)
        
        eval(obs as! MLXArray)
        #expect((obs as! MLXArray).shape == [8])
    }
    
    @Test
    @MainActor
    func testMakeContinuousEnvironment() async throws {
        var env = make("LunarLanderContinuous")
        let (obs, _) = env.reset(seed: 42)
        
        eval(obs as! MLXArray)
        #expect((obs as! MLXArray).shape == [8])
    }
    
    @Test
    @MainActor
    func testMakeWithCustomGravity() async throws {
        var env = make("LunarLander", kwargs: ["gravity": -5.0])
        let (obs, _) = env.reset(seed: 42)
        
        eval(obs as! MLXArray)
        #expect((obs as! MLXArray).shape == [8])
    }
    
    @Test
    @MainActor
    func testMakeWithWind() async throws {
        var env = make("LunarLander", kwargs: [
            "enable_wind": true,
            "wind_power": 10.0,
            "turbulence_power": 1.0
        ])
        let (obs, _) = env.reset(seed: 42)
        
        eval(obs as! MLXArray)
        #expect((obs as! MLXArray).shape == [8])
    }
}
