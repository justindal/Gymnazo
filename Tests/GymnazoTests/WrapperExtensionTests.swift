import Testing
import MLX
@testable import Gymnazo

@MainActor
@Suite("Wrapper Extension Tests")
struct WrapperExtensionTests {
    func makeCartPole() async throws -> CartPole {
        let env = try await Gymnazo.make("CartPole")
        guard let cartPole = env.unwrapped as? CartPole else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "CartPole",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return cartPole
    }

    func makePendulum() async throws -> Pendulum {
        let env = try await Gymnazo.make("Pendulum")
        guard let pendulum = env.unwrapped as? Pendulum else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "Pendulum",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return pendulum
    }
    
    @Test("orderEnforced creates OrderEnforcing wrapper")
    func testOrderEnforced() async throws {
        let baseEnv = try await makeCartPole()
        let env = baseEnv.orderEnforced()
        
        #expect(env is OrderEnforcing)
    }
    
    @Test("passiveChecked creates PassiveEnvChecker wrapper")
    func testPassiveChecked() async throws {
        let baseEnv = try await makeCartPole()
        let env = baseEnv.passiveChecked()
        
        #expect(env is PassiveEnvChecker)
    }
    
    @Test("recordingStatistics creates RecordEpisodeStatistics wrapper")
    func testRecordingStatistics() async throws {
        let baseEnv = try await makeCartPole()
        let env = try baseEnv.recordingStatistics()
        
        #expect(env is RecordEpisodeStatistics)
    }
    
    @Test("recordingStatistics with custom parameters")
    func testRecordingStatisticsCustom() async throws {
        let baseEnv = try await makeCartPole()
        let env = try baseEnv.recordingStatistics(bufferLength: 50, statsKey: "ep")
        
        #expect(env is RecordEpisodeStatistics)
    }
    
    @Test("timeLimited creates TimeLimit wrapper")
    func testTimeLimited() async throws {
        let baseEnv = try await makeCartPole()
        let env = try baseEnv.timeLimited(100)
        
        #expect(env is TimeLimit)
    }
    
    @Test("observationsNormalized creates NormalizeObservation wrapper")
    func testObservationsNormalized() async throws {
        let baseEnv = try await makeCartPole()
        let env = try baseEnv.observationsNormalized()
        
        #expect(env is NormalizeObservation)
    }
    
    @Test("observationsTransformed creates TransformObservation wrapper")
    func testObservationsTransformed() async throws {
        let baseEnv = try await makeCartPole()
        let env = baseEnv.observationsTransformed { obs in
            obs * 2
        }
        
        #expect(env is TransformObservation)
    }
    
    @Test("actionsClipped creates ClipAction wrapper")
    func testActionsClipped() async throws {
        let baseEnv = try await makePendulum()
        let env = baseEnv.actionsClipped()
        
        #expect(env is ClipAction)
    }
    
    @Test("actionsRescaled creates RescaleAction wrapper")
    func testActionsRescaled() async throws {
        let baseEnv = try await makePendulum()
        let env = baseEnv.actionsRescaled()
        
        #expect(env is RescaleAction)
    }
    
    @Test("actionsRescaled with custom range")
    func testActionsRescaledCustom() async throws {
        let baseEnv = try await makePendulum()
        let env = baseEnv.actionsRescaled(from: (low: 0.0, high: 1.0))
        
        #expect(env is RescaleAction)
    }
    
    @Test("validated creates standard validation stack")
    func testValidated() async throws {
        let baseEnv = try await makeCartPole()
        let env = baseEnv.validated()
        
        #expect(env is OrderEnforcing)
    }
    
    @Test("validated with time limit")
    func testValidatedWithTimeLimit() async throws {
        let baseEnv = try await makeCartPole()
        let env = try baseEnv.validated(maxSteps: 200)
        
        #expect(env is TimeLimit)
    }
    
    @Test("Chaining multiple wrappers")
    func testChainingWrappers() async throws {
        let env = try await makeCartPole()
            .orderEnforced()
            .recordingStatistics()
            .timeLimited(500)
        
        #expect(env is TimeLimit)
    }
    
    @Test("Chained environment works correctly")
    func testChainedEnvironmentWorks() async throws {
        let env = try await makeCartPole()
            .orderEnforced()
            .timeLimited(10)
        
        let obs = try env.reset(seed: 42, options: nil).obs
        #expect(obs.shape == [4])
        
        let result = try env.step(MLXArray(Int32(0)))
        #expect(result.obs.shape == [4])
        #expect(result.reward == 1.0)
    }
    
    @Test("Time limit wrapper truncates at limit")
    func testTimeLimitTruncates() async throws {
        let env = try await makeCartPole()
            .orderEnforced()
            .timeLimited(5)
        
        _ = try env.reset(seed: 42, options: nil)
        
        var truncated = false
        for _ in 0..<10 {
            let result = try env.step(MLXArray(Int32(0)))
            if result.truncated {
                truncated = true
                break
            }
        }
        
        #expect(truncated, "Environment should truncate after 5 steps")
    }
    
    @Test("Recording statistics tracks data")
    func testRecordingStatisticsTracksData() async throws {
        let env = try await makeCartPole()
            .orderEnforced()
            .recordingStatistics()
            .timeLimited(5)
        
        _ = try env.reset(seed: 42, options: nil)
        
        var foundEpisodeInfo = false
        for _ in 0..<10 {
            let result = try env.step(MLXArray(Int32(1)))
            if result.info["episode"] != nil {
                foundEpisodeInfo = true
                break
            }
        }
        
        #expect(foundEpisodeInfo, "Should find episode info after termination")
    }
    
    @Test("wrapped closure API")
    func testWrappedClosure() async throws {
        let baseEnv = try await makeCartPole()
        let env = baseEnv.wrapped { env in
            try! env.orderEnforced().timeLimited(100)
        }
        
        #expect(env is TimeLimit)
    }
    
    @Test("Observation transform applies correctly")
    func testObservationTransformApplies() async throws {
        let baseEnv = try await makeCartPole()
        var env = baseEnv.observationsTransformed { obs in
            obs * 2
        }
        
        let obs = try env.reset(seed: 42, options: nil).obs
        
        var comparisonEnv = try await makeCartPole()
        let baseObs = try comparisonEnv.reset(seed: 42, options: nil).obs
        
        let obsValues: [Float] = obs.asArray(Float.self)
        let baseValues: [Float] = baseObs.asArray(Float.self)
        
        for i in 0..<obsValues.count {
            #expect(abs(obsValues[i] - baseValues[i] * 2) < 1e-5)
        }
    }
    
    @Test("Normalized observations have reasonable values")
    func testNormalizedObservationsReasonable() async throws {
        let baseEnv = try await makeCartPole()
        var env = try baseEnv.observationsNormalized()
        
        _ = try env.reset(seed: 42, options: nil)
        
        for _ in 0..<100 {
            let result = try env.step(MLXArray(Int32.random(in: 0..<2)))
            let values: [Float] = result.obs.asArray(Float.self)
            
            for value in values {
                #expect(abs(value) < 100, "Normalized values should be bounded")
            }
            
            if result.terminated || result.truncated {
                _ = try env.reset(seed: nil, options: nil)
            }
        }
    }
}
