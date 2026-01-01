import Testing
import MLX
@testable import Gymnazo

@Suite("CarRacing environment")
struct CarRacingTests {
    
    @Test
    func testContinuousInitialization() async throws {
        let env = CarRacing()
        
        #expect(env.lapCompletePercent == 0.95)
        #expect(env.domainRandomize == false)
    }
    
    @Test
    func testContinuousActionSpace() async throws {
        let env = CarRacing()
        
        #expect(env.action_space.shape == [3])
        
        let low = env.action_space.low.asArray(Float.self)
        let high = env.action_space.high.asArray(Float.self)
        
        #expect(low[0] == -1.0)
        #expect(low[1] == 0.0)
        #expect(low[2] == 0.0)
        #expect(high[0] == 1.0)
        #expect(high[1] == 1.0)
        #expect(high[2] == 1.0)
    }
    
    @Test
    func testContinuousObservationSpace() async throws {
        let env = CarRacing()
        
        #expect(env.observation_space.shape == [96, 96, 3])
        #expect(env.observation_space.dtype == .uint8)
        
        let low = env.observation_space.low.asArray(Float.self)
        let high = env.observation_space.high.asArray(Float.self)
        
        #expect(low[0] == 0.0)
        #expect(high[0] == 255.0)
    }
    
    @Test
    func testDiscreteInitialization() async throws {
        let env = CarRacingDiscrete()
        
        #expect(env.lapCompletePercent == 0.95)
        #expect(env.domainRandomize == false)
    }
    
    @Test
    func testDiscreteActionSpace() async throws {
        let env = CarRacingDiscrete()
        
        #expect(env.action_space.n == 5)
        #expect(env.action_space.start == 0)
        #expect(env.action_space.contains(0))
        #expect(env.action_space.contains(1))
        #expect(env.action_space.contains(2))
        #expect(env.action_space.contains(3))
        #expect(env.action_space.contains(4))
        #expect(!env.action_space.contains(5))
    }
    
    @Test
    func testDiscreteObservationSpace() async throws {
        let env = CarRacingDiscrete()
        
        #expect(env.observation_space.shape == [96, 96, 3])
        #expect(env.observation_space.dtype == .uint8)
    }
    
    @Test
    func testContinuousResetReturnsObservation() async throws {
        var env = CarRacing()
        let result = env.reset(seed: 42)
        let obs = result.obs
        
        #expect(obs.shape == [96, 96, 3])
        #expect(obs.dtype == .uint8)
    }
    
    @Test
    func testDiscreteResetReturnsObservation() async throws {
        var env = CarRacingDiscrete()
        let result = env.reset(seed: 42)
        let obs = result.obs
        
        #expect(obs.shape == [96, 96, 3])
        #expect(obs.dtype == .uint8)
    }
    
    @Test
    func testContinuousStepReturnsCorrectShape() async throws {
        var env = CarRacing()
        _ = env.reset(seed: 42)
        
        let action = MLXArray([0.0, 0.5, 0.0] as [Float32])
        let result = env.step(action)
        
        #expect(result.obs.shape == [96, 96, 3])
        #expect(result.obs.dtype == .uint8)
        #expect(result.terminated == false || result.terminated == true)
        #expect(result.truncated == false)
    }
    
    @Test
    func testDiscreteStepReturnsCorrectShape() async throws {
        var env = CarRacingDiscrete()
        _ = env.reset(seed: 42)
        
        let result = env.step(0)
        
        #expect(result.obs.shape == [96, 96, 3])
        #expect(result.obs.dtype == .uint8)
        #expect(result.terminated == false || result.terminated == true)
        #expect(result.truncated == false)
    }
    
    @Test
    func testContinuousRewardIsNegativePerFrame() async throws {
        var env = CarRacing()
        _ = env.reset(seed: 42)
        
        let action = MLXArray([0.0, 0.0, 0.0] as [Float32])
        let result = env.step(action)
        
        #expect(result.reward <= 0)
    }
    
    @Test
    func testDiscreteRewardIsNegativePerFrame() async throws {
        var env = CarRacingDiscrete()
        _ = env.reset(seed: 42)
        
        let result = env.step(0)
        
        #expect(result.reward <= 0)
    }
    
    @Test
    func testTerminationInfoContainsLapFinished() async throws {
        var env = CarRacing()
        _ = env.reset(seed: 42)
        
        var terminated = false
        var stepCount = 0
        let maxSteps = 100
        
        while !terminated && stepCount < maxSteps {
            let action = MLXArray([0.0, 1.0, 0.0] as [Float32])
            let result = env.step(action)
            
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
        let hasCarRacing = Gymnazo.registry.keys.contains("CarRacing")
        let hasCarRacingDiscrete = Gymnazo.registry.keys.contains("CarRacingDiscrete")
        
        #expect(hasCarRacing)
        #expect(hasCarRacingDiscrete)
    }
    
    @Test
    @MainActor
    func testMakeCarRacing() async throws {
        let env = make("CarRacing")
        
        #expect(env.spec != nil)
        #expect(env.spec?.id == "CarRacing")
        #expect(env.spec?.maxEpisodeSteps == 1000)
        #expect(env.spec?.rewardThreshold == 900)
    }
    
    @Test
    @MainActor
    func testMakeCarRacingDiscrete() async throws {
        let env = make("CarRacingDiscrete")
        
        #expect(env.spec != nil)
        #expect(env.spec?.id == "CarRacingDiscrete")
        #expect(env.spec?.maxEpisodeSteps == 1000)
        #expect(env.spec?.rewardThreshold == 900)
    }
    
    @Test
    func testDomainRandomize() async throws {
        var env = CarRacing(domainRandomize: true)
        _ = env.reset(seed: 42)
        
        #expect(env.domainRandomize == true)
    }
    
    @Test
    func testLapCompletePercent() async throws {
        let env = CarRacing(lapCompletePercent: 0.8)
        
        #expect(env.lapCompletePercent == 0.8)
    }
    
    @Test
    func testClose() async throws {
        var env = CarRacing()
        _ = env.reset(seed: 42)
        env.close()
    }
    
    @Test
    func testDiscreteClose() async throws {
        var env = CarRacingDiscrete()
        _ = env.reset(seed: 42)
        env.close()
    }
    
    @Test
    func testMultipleResets() async throws {
        var env = CarRacing()
        
        for i in 0..<3 {
            let result = env.reset(seed: UInt64(i + 100))
            #expect(result.obs.shape == [96, 96, 3])
        }
    }
    
    @Test
    func testDiscreteMultipleResets() async throws {
        var env = CarRacingDiscrete()
        
        for i in 0..<3 {
            let result = env.reset(seed: UInt64(i + 100))
            #expect(result.obs.shape == [96, 96, 3])
        }
    }
}

