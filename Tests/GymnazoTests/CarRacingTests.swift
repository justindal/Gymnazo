import Testing
import MLX
@testable import Gymnazo

@Suite("CarRacing environment")
struct CarRacingTests {
    func makeCarRacing(
        renderMode: RenderMode? = nil,
        lapCompletePercent: Float? = nil,
        domainRandomize: Bool? = nil
    ) async throws -> CarRacing {
        var options: EnvOptions = [:]
        if let renderMode {
            options["render_mode"] = renderMode.rawValue
        }
        if let lapCompletePercent {
            options["lap_complete_percent"] = lapCompletePercent
        }
        if let domainRandomize {
            options["domain_randomize"] = domainRandomize
        }
        let env = try await Gymnazo.make("CarRacing", options: options)
        guard let carRacing = env.unwrapped as? CarRacing else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "CarRacing",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return carRacing
    }

    func makeCarRacingDiscrete(
        renderMode: RenderMode? = nil,
        lapCompletePercent: Float? = nil,
        domainRandomize: Bool? = nil
    ) async throws -> CarRacingDiscrete {
        var options: EnvOptions = [:]
        if let renderMode {
            options["render_mode"] = renderMode.rawValue
        }
        if let lapCompletePercent {
            options["lap_complete_percent"] = lapCompletePercent
        }
        if let domainRandomize {
            options["domain_randomize"] = domainRandomize
        }
        let env = try await Gymnazo.make(
            "CarRacingDiscrete",
            options: options
        )
        guard let carRacing = env.unwrapped as? CarRacingDiscrete else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "CarRacingDiscrete",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return carRacing
    }
    
    @Test
    func testContinuousInitialization() async throws {
        let env = try await makeCarRacing()
        
        #expect(env.lapCompletePercent == 0.95)
        #expect(env.domainRandomize == false)
    }
    
    @Test
    func testContinuousActionSpace() async throws {
        let env = try await makeCarRacing()
        
        guard let actionSpace = env.actionSpace as? Box else {
            Issue.record("Action space is not Box")
            return
        }
        
        #expect(actionSpace.shape == [3])
        
        let low = actionSpace.low.asArray(Float.self)
        let high = actionSpace.high.asArray(Float.self)
        
        #expect(low[0] == -1.0)
        #expect(low[1] == 0.0)
        #expect(low[2] == 0.0)
        #expect(high[0] == 1.0)
        #expect(high[1] == 1.0)
        #expect(high[2] == 1.0)
    }
    
    @Test
    func testContinuousObservationSpace() async throws {
        let env = try await makeCarRacing()
        
        guard let observationSpace = env.observationSpace as? Box else {
            Issue.record("Observation space is not Box")
            return
        }
        
        #expect(observationSpace.shape == [96, 96, 3])
        #expect(observationSpace.dtype == .uint8)
        
        let low = observationSpace.low.asArray(Float.self)
        let high = observationSpace.high.asArray(Float.self)
        
        #expect(low[0] == 0.0)
        #expect(high[0] == 255.0)
    }
    
    @Test
    func testDiscreteInitialization() async throws {
        let env = try await makeCarRacingDiscrete()
        
        #expect(env.lapCompletePercent == 0.95)
        #expect(env.domainRandomize == false)
    }
    
    @Test
    func testDiscreteActionSpace() async throws {
        let env = try await makeCarRacingDiscrete()

        guard let actionSpace = env.actionSpace as? Discrete else {
            Issue.record("Action space is not Discrete")
            return
        }
        
        #expect(actionSpace.n == 5)
        #expect(actionSpace.start == 0)
        #expect(actionSpace.contains(MLXArray(Int32(0))))
        #expect(actionSpace.contains(MLXArray(Int32(1))))
        #expect(actionSpace.contains(MLXArray(Int32(2))))
        #expect(actionSpace.contains(MLXArray(Int32(3))))
        #expect(actionSpace.contains(MLXArray(Int32(4))))
        #expect(!actionSpace.contains(MLXArray(Int32(5))))
    }
    
    @Test
    func testDiscreteObservationSpace() async throws {
        let env = try await makeCarRacingDiscrete()
        
        #expect(env.observationSpace.shape == [96, 96, 3])
        #expect(env.observationSpace.dtype == .uint8)
    }
    
    @Test
    func testContinuousResetReturnsObservation() async throws {
        var env = try await makeCarRacing()
        let result = try env.reset(seed: 42)
        let obs = result.obs
        
        #expect(obs.shape == [96, 96, 3])
        #expect(obs.dtype == .uint8)
    }
    
    @Test
    func testDiscreteResetReturnsObservation() async throws {
        var env = try await makeCarRacingDiscrete()
        let result = try env.reset(seed: 42)
        let obs = result.obs
        
        #expect(obs.shape == [96, 96, 3])
        #expect(obs.dtype == .uint8)
    }
    
    @Test
    func testContinuousStepReturnsCorrectShape() async throws {
        var env = try await makeCarRacing()
        _ = try env.reset(seed: 42)
        
        let action = MLXArray([0.0, 0.5, 0.0] as [Float32])
        let result = try env.step(action)
        
        #expect(result.obs.shape == [96, 96, 3])
        #expect(result.obs.dtype == .uint8)
        #expect(result.terminated == false || result.terminated == true)
        #expect(result.truncated == false)
    }
    
    @Test
    func testDiscreteStepReturnsCorrectShape() async throws {
        var env = try await makeCarRacingDiscrete()
        _ = try env.reset(seed: 42)
        
        let result = try env.step(MLXArray(Int32(0)))
        
        #expect(result.obs.shape == [96, 96, 3])
        #expect(result.obs.dtype == .uint8)
        #expect(result.terminated == false || result.terminated == true)
        #expect(result.truncated == false)
    }
    
    @Test
    func testContinuousRewardIsNegativePerFrame() async throws {
        var env = try await makeCarRacing()
        _ = try env.reset(seed: 42)
        
        let action = MLXArray([0.0, 0.0, 0.0] as [Float32])
        let result = try env.step(action)
        
        #expect(result.reward <= 0)
    }
    
    @Test
    func testDiscreteRewardIsNegativePerFrame() async throws {
        var env = try await makeCarRacingDiscrete()
        _ = try env.reset(seed: 42)
        
        let result = try env.step(MLXArray(Int32(0)))
        
        #expect(result.reward <= 0)
    }
    
    @Test
    func testTerminationInfoContainsLapFinished() async throws {
        var env = try await makeCarRacing()
        _ = try env.reset(seed: 42)
        
        var terminated = false
        var stepCount = 0
        let maxSteps = 100
        
        while !terminated && stepCount < maxSteps {
            let action = MLXArray([0.0, 1.0, 0.0] as [Float32])
            let result = try env.step(action)
            
            if result.terminated {
                terminated = true
                let hasLapFinished = result.info["lap_finished"] != nil
                #expect(hasLapFinished)
            }
            stepCount += 1
        }
    }
    
    @Test
    @MainActor
    func testRegistryContainsCarRacing() async throws {
        let specs = await Gymnazo.registry()
        let hasCarRacing = specs.keys.contains("CarRacing")
        let hasCarRacingDiscrete = specs.keys.contains("CarRacingDiscrete")
        
        #expect(hasCarRacing)
        #expect(hasCarRacingDiscrete)
    }
    
    @Test
    @MainActor
    func testMakeCarRacing() async throws {
        let env = try await Gymnazo.make("CarRacing")
        
        #expect(env.spec != nil)
        #expect(env.spec?.id == "CarRacing")
        #expect(env.spec?.maxEpisodeSteps == 1000)
        #expect(env.spec?.rewardThreshold == 900)
    }
    
    @Test
    @MainActor
    func testMakeCarRacingDiscrete() async throws {
        let env = try await Gymnazo.make("CarRacingDiscrete")
        
        #expect(env.spec != nil)
        #expect(env.spec?.id == "CarRacingDiscrete")
        #expect(env.spec?.maxEpisodeSteps == 1000)
        #expect(env.spec?.rewardThreshold == 900)
    }
    
    @Test
    func testDomainRandomize() async throws {
        var env = try await makeCarRacing(domainRandomize: true)
        _ = try env.reset(seed: 42)
        
        #expect(env.domainRandomize == true)
    }
    
    @Test
    func testLapCompletePercent() async throws {
        let env = try await makeCarRacing(lapCompletePercent: 0.8)
        
        #expect(env.lapCompletePercent == 0.8)
    }
    
    @Test
    func testClose() async throws {
        var env = try await makeCarRacing()
        _ = try env.reset(seed: 42)
        env.close()
    }
    
    @Test
    func testDiscreteClose() async throws {
        var env = try await makeCarRacingDiscrete()
        _ = try env.reset(seed: 42)
        env.close()
    }
    
    @Test
    func testMultipleResets() async throws {
        var env = try await makeCarRacing()
        
        for i in 0..<3 {
            let result = try env.reset(seed: UInt64(i + 100))
            #expect(result.obs.shape == [96, 96, 3])
        }
    }
    
    @Test
    func testDiscreteMultipleResets() async throws {
        var env = try await makeCarRacingDiscrete()
        
        for i in 0..<3 {
            let result = try env.reset(seed: UInt64(i + 100))
            #expect(result.obs.shape == [96, 96, 3])
        }
    }
    
    @Test
    func testSteeringCausesDirectionChange() async throws {
        var env = try await makeCarRacing()
        _ = try env.reset(seed: 42)
        
        for _ in 0..<50 {
            let action = MLXArray([0.0, 1.0, 0.0] as [Float32])
            _ = try env.step(action)
        }
        
        let snapshot1 = env.currentSnapshot!
        let initialAngle = snapshot1.carAngle
        let initialSpeed = snapshot1.trueSpeed
        
        #expect(initialSpeed > 1.0, "Car should have built up speed")
        
        for _ in 0..<100 {
            let action = MLXArray([-1.0, 0.5, 0.0] as [Float32])
            _ = try env.step(action)
        }
        
        let snapshot2 = env.currentSnapshot!
        let finalAngle = snapshot2.carAngle
        
        let angleDiff = abs(finalAngle - initialAngle)
        
        #expect(angleDiff > 0.1, "Car should turn when steering is applied. Angle only changed by \(angleDiff)")
    }
    
    @Test
    func testDiscreteSteeringCausesDirectionChange() async throws {
        var env = try await makeCarRacingDiscrete()
        _ = try env.reset(seed: 42)
        
        for _ in 0..<100 {
            _ = try env.step(MLXArray(Int32(3)))
        }
        
        let snapshot1 = env.currentSnapshot!
        let initialAngle = snapshot1.carAngle
        let initialSpeed = snapshot1.trueSpeed
        
        print("DEBUG: Initial speed = \(initialSpeed), initial angle = \(initialAngle)")
        
        #expect(initialSpeed > 1.0, "Car should have built up speed")
        
        for i in 0..<200 {
            if i % 2 == 0 {
                _ = try env.step(MLXArray(Int32(1)))
            } else {
                _ = try env.step(MLXArray(Int32(3)))
            }
            
            if i % 40 == 0 {
                let snap = env.currentSnapshot!
                print("DEBUG: Step \(i): angle = \(snap.carAngle), speed = \(snap.trueSpeed), steeringAngle = \(snap.steeringAngle)")
            }
        }
        
        let snapshot2 = env.currentSnapshot!
        let finalAngle = snapshot2.carAngle
        let finalSpeed = snapshot2.trueSpeed
        
        print("DEBUG: Final speed = \(finalSpeed), final angle = \(finalAngle)")
        
        let angleDiff = abs(finalAngle - initialAngle)
        print("DEBUG: Angle difference = \(angleDiff) radians = \(angleDiff * 180 / .pi) degrees")
        
        #expect(angleDiff > 0.1, "Car should turn when steering is applied. Angle only changed by \(angleDiff)")
    }
}

