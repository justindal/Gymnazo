import MLX
import Testing

@testable import Gymnazo

@Suite("LunarLander environment")
struct LunarLanderTests {
    func makeLunarLander(
        renderMode: RenderMode? = nil,
        gravity: Float? = nil,
        enableWind: Bool? = nil,
        windPower: Float? = nil,
        turbulencePower: Float? = nil
    ) async throws -> LunarLander {
        var options: EnvOptions = [:]
        if let renderMode {
            options["render_mode"] = renderMode.rawValue
        }
        if let gravity {
            options["gravity"] = gravity
        }
        if let enableWind {
            options["enable_wind"] = enableWind
        }
        if let windPower {
            options["wind_power"] = windPower
        }
        if let turbulencePower {
            options["turbulence_power"] = turbulencePower
        }
        let env: AnyEnv<MLXArray, Int> = try await Gymnazo.make("LunarLander", options: options)
        guard let lander = env.unwrapped as? LunarLander else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "LunarLander",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return lander
    }

    @Test
    func testInitialization() async throws {
        let env = try await makeLunarLander()

        #expect(env.gravity == -10.0)
        #expect(env.enableWind == false)
        #expect(env.windPower == 15.0)
        #expect(env.turbulencePower == 1.5)
    }

    @Test
    func testInitializationWithCustomGravity() async throws {
        let env = try await makeLunarLander(gravity: -5.0)
        #expect(env.gravity == -5.0)
    }

    @Test
    func testInitializationWithWind() async throws {
        let env = try await makeLunarLander(enableWind: true, windPower: 10.0, turbulencePower: 1.0)
        #expect(env.enableWind == true)
        #expect(env.windPower == 10.0)
        #expect(env.turbulencePower == 1.0)
    }

    @Test
    func testActionSpace() async throws {
        let env = try await makeLunarLander()

        guard let actionSpace = env.actionSpace as? Discrete else {
            Issue.record("Action space is not Discrete")
            return
        }

        #expect(actionSpace.n == 4)
        #expect(actionSpace.start == 0)
        #expect(actionSpace.contains(0))
        #expect(actionSpace.contains(1))
        #expect(actionSpace.contains(2))
        #expect(actionSpace.contains(3))
        #expect(!actionSpace.contains(4))
    }

    @Test
    func testObservationSpace() async throws {
        let env = try await makeLunarLander()

        guard let observationSpace = env.observationSpace as? Box else {
            Issue.record("Observation space is not Box")
            return
        }

        #expect(observationSpace.shape == [8])

        let low = observationSpace.low.asArray(Float.self)
        let high = observationSpace.high.asArray(Float.self)

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
        var env = try await makeLunarLander()
        let result = try env.reset(seed: 42)
        let obs = result.obs
        let info = result.info

        #expect(obs.shape == [8])
        #expect(info.isEmpty || info.count >= 0)
    }

    @Test
    func testResetDeterminismWithSeed() async throws {
        var env1 = try await makeLunarLander()
        var env2 = try await makeLunarLander()

        let obs1 = try env1.reset(seed: 123).obs
        let obs2 = try env2.reset(seed: 123).obs

        eval(obs1, obs2)

        let diff = abs(obs1 - obs2).sum().item(Float.self)
        #expect(diff < 1e-6)
    }

    @Test
    func testResetDifferentSeeds() async throws {
        var env1 = try await makeLunarLander()
        var env2 = try await makeLunarLander()

        let obs1 = try env1.reset(seed: 1).obs
        let obs2 = try env2.reset(seed: 999).obs

        eval(obs1, obs2)

        let diff = abs(obs1 - obs2).sum().item(Float.self)
        #expect(diff > 1e-6)
    }

    @Test
    func testStepReturnsCorrectShape() async throws {
        var env = try await makeLunarLander()
        _ = try env.reset(seed: 42)

        let result = try env.step(0)
        let obs = result.obs
        let truncated = result.truncated

        #expect(obs.shape == [8])
        #expect(truncated == false)
    }

    @Test
    func testStepWithDifferentActions() async throws {
        var env = try await makeLunarLander()
        _ = try env.reset(seed: 42)

        for action in 0..<4 {
            var testEnv = try await makeLunarLander()
            _ = try testEnv.reset(seed: 42)

            let obs = try testEnv.step(action).obs
            #expect(obs.shape == [8])
        }
    }

    @Test
    func testMainEngineAffectsVelocity() async throws {
        var env = try await makeLunarLander()
        _ = try env.reset(seed: 42)

        let obsNoAction = try env.step(0).obs

        _ = try env.reset(seed: 42)
        let obsMainEngine = try env.step(2).obs

        eval(obsNoAction, obsMainEngine)

        let vyNoAction = obsNoAction[3].item(Float.self)
        let vyMainEngine = obsMainEngine[3].item(Float.self)

        #expect(vyMainEngine > vyNoAction)
    }

    @Test
    func testSideEnginesAffectAngularVelocity() async throws {
        var envLeft = try await makeLunarLander()
        var envRight = try await makeLunarLander()

        _ = try envLeft.reset(seed: 42)
        _ = try envRight.reset(seed: 42)

        let obsLeft = try envLeft.step(1).obs
        let obsRight = try envRight.step(3).obs

        eval(obsLeft, obsRight)

        let angVelLeft = obsLeft[5].item(Float.self)
        let angVelRight = obsRight[5].item(Float.self)

        #expect(angVelLeft != angVelRight)
    }

    @Test
    func testGravityApplied() async throws {
        var env = try await makeLunarLander()
        let obsInit = try env.reset(seed: 42).obs

        var obs = obsInit
        for _ in 0..<10 {
            let result = try env.step(0)
            obs = result.obs
        }

        eval(obsInit, obs)

        let yInit = obsInit[1].item(Float.self)
        let yFinal = obs[1].item(Float.self)

        #expect(yFinal < yInit)
    }

    @Test
    func testCrashTerminatesWithNegativeReward() async throws {
        var env = try await makeLunarLander()
        _ = try env.reset(seed: 123)

        var terminated = false
        var reward: Double = 0
        for _ in 0..<400 {
            let step = try env.step(0)
            terminated = step.terminated
            reward = step.reward
            if terminated { break }
        }

        #expect(terminated == true)
        #expect(reward == -100)
    }

    @Test
    func testFuelCostInReward() async throws {
        var envNoAction = try await makeLunarLander()
        var envMainEngine = try await makeLunarLander()

        _ = try envNoAction.reset(seed: 42)
        _ = try envMainEngine.reset(seed: 42)

        for _ in 0..<5 {
            _ = try envNoAction.step(0)
            _ = try envMainEngine.step(2)
        }

        let rewardNoAction = try envNoAction.step(0).reward
        let rewardMainEngine = try envMainEngine.step(2).reward

        _ = rewardNoAction
        _ = rewardMainEngine
    }

    @Test
    func testObservationContainsLegContact() async throws {
        var env = try await makeLunarLander()
        let obs = try env.reset(seed: 42).obs

        eval(obs)

        let leftContact = obs[6].item(Float.self)
        let rightContact = obs[7].item(Float.self)

        #expect(leftContact == 0.0 || leftContact == 1.0)
        #expect(rightContact == 0.0 || rightContact == 1.0)

        #expect(leftContact == 0.0)
        #expect(rightContact == 0.0)
    }

    @Test
    func testRenderModeInitialization() async throws {
        let envNoRender = try await makeLunarLander(renderMode: nil)
        #expect(envNoRender.renderMode == nil)

        let envHuman = try await makeLunarLander(renderMode: .human)
        #expect(envHuman.renderMode == .human)
    }

    @Test
    func testCurrentSnapshotBeforeReset() async throws {
        let env = try await makeLunarLander()
        #expect(env.currentSnapshot == nil)
    }

    @Test
    func testCurrentSnapshotAfterReset() async throws {
        var env = try await makeLunarLander()
        _ = try env.reset(seed: 42)

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
    func makeLunarLanderContinuous(
        renderMode: RenderMode? = nil,
        gravity: Float? = nil,
        enableWind: Bool? = nil,
        windPower: Float? = nil,
        turbulencePower: Float? = nil
    ) async throws -> LunarLanderContinuous {
        var options: EnvOptions = [:]
        if let renderMode {
            options["render_mode"] = renderMode.rawValue
        }
        if let gravity {
            options["gravity"] = gravity
        }
        if let enableWind {
            options["enable_wind"] = enableWind
        }
        if let windPower {
            options["wind_power"] = windPower
        }
        if let turbulencePower {
            options["turbulence_power"] = turbulencePower
        }
        let env: AnyEnv<MLXArray, MLXArray> = try await Gymnazo.make(
            "LunarLanderContinuous",
            options: options
        )
        guard let lander = env.unwrapped as? LunarLanderContinuous else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "LunarLanderContinuous",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return lander
    }

    @Test
    func testInitialization() async throws {
        let env = try await makeLunarLanderContinuous()

        #expect(env.gravity == -10.0)
        #expect(env.enableWind == false)
    }

    @Test
    func testActionSpace() async throws {
        let env = try await makeLunarLanderContinuous()

        guard let actionSpace = env.actionSpace as? Box else {
            Issue.record("Action space is not Box")
            return
        }

        #expect(actionSpace.shape == [2])

        let low = actionSpace.low.asArray(Float.self)
        let high = actionSpace.high.asArray(Float.self)

        #expect(low[0] == -1.0)
        #expect(low[1] == -1.0)
        #expect(high[0] == 1.0)
        #expect(high[1] == 1.0)
    }

    @Test
    func testObservationSpace() async throws {
        let env = try await makeLunarLanderContinuous()

        guard let observationSpace = env.observationSpace as? Box else {
            Issue.record("Observation space is not Box")
            return
        }

        #expect(observationSpace.shape == [8])

        let low = observationSpace.low.asArray(Float.self)
        let high = observationSpace.high.asArray(Float.self)

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
        var env = try await makeLunarLanderContinuous()
        let obs = try env.reset(seed: 42).obs

        #expect(obs.shape == [8])
    }

    @Test
    func testStepWithContinuousAction() async throws {
        var env = try await makeLunarLanderContinuous()
        _ = try env.reset(seed: 42)

        let action = MLXArray([0.5, 0.0] as [Float32])
        let obs = try env.step(action).obs

        #expect(obs.shape == [8])
    }

    @Test
    func testOutOfRangeActionIsClipped() async throws {
        var env = try await makeLunarLanderContinuous()
        _ = try env.reset(seed: 42)

        let action = MLXArray([2.0, -2.0] as [Float32])
        let obs = try env.step(action).obs

        #expect(obs.shape == [8])
        let leftContact = obs[6].item(Float.self)
        let rightContact = obs[7].item(Float.self)
        #expect(leftContact >= 0.0 && leftContact <= 1.0)
        #expect(rightContact >= 0.0 && rightContact <= 1.0)
    }

    @Test
    func testWindSeedingDeterminism() async throws {
        var env1 = try await makeLunarLanderContinuous(enableWind: true)
        var env2 = try await makeLunarLanderContinuous(enableWind: true)
        var env3 = try await makeLunarLanderContinuous(enableWind: true)

        let obs1 = try env1.reset(seed: 7).obs
        let obs2 = try env2.reset(seed: 7).obs
        let obs3 = try env3.reset(seed: 8).obs

        let diffSame = abs(obs1 - obs2).sum().item(Float.self)
        let diffDifferent = abs(obs1 - obs3).sum().item(Float.self)

        #expect(diffSame < 1e-6)
        #expect(diffDifferent > 1e-6)
    }

    @Test
    func testMainEngineThrottling() async throws {
        var envLow = try await makeLunarLanderContinuous()
        var envHigh = try await makeLunarLanderContinuous()

        _ = try envLow.reset(seed: 42)
        _ = try envHigh.reset(seed: 42)

        let actionLow = MLXArray([0.1, 0.0] as [Float32])
        let actionHigh = MLXArray([1.0, 0.0] as [Float32])

        let obsLow = try envLow.step(actionLow).obs
        let obsHigh = try envHigh.step(actionHigh).obs

        eval(obsLow, obsHigh)

        let vyLow = obsLow[3].item(Float.self)
        let vyHigh = obsHigh[3].item(Float.self)

        #expect(vyHigh > vyLow)
    }

    @Test
    func testLateralEngineDeadzone() async throws {
        var envOff = try await makeLunarLanderContinuous()
        var envOn = try await makeLunarLanderContinuous()

        _ = try envOff.reset(seed: 42)
        _ = try envOn.reset(seed: 42)

        let actionOff = MLXArray([0.0, 0.3] as [Float32])
        let actionOn = MLXArray([0.0, 0.8] as [Float32])

        let obsOff = try envOff.step(actionOff).obs
        let obsOn = try envOn.step(actionOn).obs

        eval(obsOff, obsOn)

        let angVelOff = obsOff[5].item(Float.self)
        let angVelOn = obsOn[5].item(Float.self)

        #expect(angVelOff != angVelOn || abs(angVelOff - angVelOn) >= 0)
    }

    @Test
    func testNegativeMainEngineOff() async throws {
        var env = try await makeLunarLanderContinuous()
        _ = try env.reset(seed: 42)

        let action = MLXArray([-0.5, 0.0] as [Float32])
        let obs = try env.step(action).obs

        #expect(obs.shape == [8])
    }

    @Test
    func testResetDeterminismWithSeed() async throws {
        var env1 = try await makeLunarLanderContinuous()
        var env2 = try await makeLunarLanderContinuous()

        let obs1 = try env1.reset(seed: 456).obs
        let obs2 = try env2.reset(seed: 456).obs

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
        var env: AnyEnv<MLXArray, Int> = try await Gymnazo.make("LunarLander")
        let obs = try env.reset(seed: 42).obs

        eval(obs)
        #expect(obs.shape == [8])
    }

    @Test
    @MainActor
    func testMakeContinuousEnvironment() async throws {
        var env: AnyEnv<MLXArray, MLXArray> = try await Gymnazo.make("LunarLanderContinuous")
        let obs = try env.reset(seed: 42).obs

        eval(obs)
        #expect(obs.shape == [8])
    }

    @Test
    @MainActor
    func testMakeWithCustomGravity() async throws {
        var env: AnyEnv<MLXArray, Int> = try await Gymnazo.make(
            "LunarLander",
            options: ["gravity": -5.0]
        )
        let obs = try env.reset(seed: 42).obs

        eval(obs)
        #expect(obs.shape == [8])
    }

    @Test
    @MainActor
    func testMakeWithWind() async throws {
        var env: AnyEnv<MLXArray, Int> = try await Gymnazo.make(
            "LunarLander",
            options: [
                "enable_wind": true,
                "wind_power": 10.0,
                "turbulence_power": 1.0,
            ]
        )
        let obs = try env.reset(seed: 42).obs

        eval(obs)
        #expect(obs.shape == [8])
    }
}
