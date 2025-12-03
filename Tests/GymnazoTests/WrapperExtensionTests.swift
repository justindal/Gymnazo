//
//  WrapperExtensionTests.swift
//

import Testing
import MLX
@testable import Gymnazo

@MainActor
@Suite("Wrapper Extension Tests")
struct WrapperExtensionTests {
    
    @Test("orderEnforced creates OrderEnforcing wrapper")
    func testOrderEnforced() {
        let env = CartPole().orderEnforced()
        
        #expect(env is OrderEnforcing<CartPole>)
    }
    
    @Test("passiveChecked creates PassiveEnvChecker wrapper")
    func testPassiveChecked() {
        let env = CartPole().passiveChecked()
        
        #expect(env is PassiveEnvChecker<CartPole>)
    }
    
    @Test("recordingStatistics creates RecordEpisodeStatistics wrapper")
    func testRecordingStatistics() {
        let env = CartPole().recordingStatistics()
        
        #expect(env is RecordEpisodeStatistics<CartPole>)
    }
    
    @Test("recordingStatistics with custom parameters")
    func testRecordingStatisticsCustom() {
        let env = CartPole().recordingStatistics(bufferLength: 50, statsKey: "ep")
        
        #expect(env is RecordEpisodeStatistics<CartPole>)
    }
    
    @Test("timeLimited creates TimeLimit wrapper")
    func testTimeLimited() {
        let env = CartPole().timeLimited(100)
        
        #expect(env is TimeLimit<CartPole>)
    }
    
    @Test("observationsNormalized creates NormalizeObservation wrapper")
    func testObservationsNormalized() {
        let env = CartPole().observationsNormalized()
        
        #expect(env is NormalizeObservation<CartPole>)
    }
    
    @Test("observationsTransformed creates TransformObservation wrapper")
    func testObservationsTransformed() {
        let env = CartPole().observationsTransformed { obs in
            obs * 2
        }
        
        #expect(env is TransformObservation<CartPole>)
    }
    
    @Test("actionsClipped creates ClipAction wrapper")
    func testActionsClipped() {
        let env = Pendulum().actionsClipped()
        
        #expect(env is ClipAction<Pendulum>)
    }
    
    @Test("actionsRescaled creates RescaleAction wrapper")
    func testActionsRescaled() {
        let env = Pendulum().actionsRescaled()
        
        #expect(env is RescaleAction<Pendulum>)
    }
    
    @Test("actionsRescaled with custom range")
    func testActionsRescaledCustom() {
        let env = Pendulum().actionsRescaled(from: (low: 0.0, high: 1.0))
        
        #expect(env is RescaleAction<Pendulum>)
    }
    
    @Test("validated creates standard validation stack")
    func testValidated() {
        let env = CartPole().validated()
        
        #expect(env is OrderEnforcing<PassiveEnvChecker<CartPole>>)
    }
    
    @Test("validated with time limit")
    func testValidatedWithTimeLimit() {
        let env = CartPole().validated(maxSteps: 200)
        
        #expect(env is TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPole>>>)
    }
    
    @Test("Chaining multiple wrappers")
    func testChainingWrappers() {
        let env = CartPole()
            .orderEnforced()
            .recordingStatistics()
            .timeLimited(500)
        
        #expect(env is TimeLimit<RecordEpisodeStatistics<OrderEnforcing<CartPole>>>)
    }
    
    @Test("Chained environment works correctly")
    func testChainedEnvironmentWorks() {
        var env = CartPole()
            .orderEnforced()
            .timeLimited(10)
        
        let (obs, _) = env.reset(seed: 42, options: nil)
        #expect(obs.shape == [4])
        
        let result = env.step(0)
        #expect(result.obs.shape == [4])
        #expect(result.reward == 1.0)
    }
    
    @Test("Time limit wrapper truncates at limit")
    func testTimeLimitTruncates() {
        var env = CartPole()
            .orderEnforced()
            .timeLimited(5)
        
        _ = env.reset(seed: 42, options: nil)
        
        var truncated = false
        for _ in 0..<10 {
            let result = env.step(0)
            if result.truncated {
                truncated = true
                break
            }
        }
        
        #expect(truncated, "Environment should truncate after 5 steps")
    }
    
    @Test("Recording statistics tracks data")
    func testRecordingStatisticsTracksData() {
        var env = CartPole()
            .orderEnforced()
            .recordingStatistics()
            .timeLimited(5)
        
        _ = env.reset(seed: 42, options: nil)
        
        var foundEpisodeInfo = false
        for _ in 0..<10 {
            let result = env.step(1)
            if result.info["episode"] != nil {
                foundEpisodeInfo = true
                break
            }
        }
        
        #expect(foundEpisodeInfo, "Should find episode info after termination")
    }
    
    @Test("wrapped closure API")
    func testWrappedClosure() {
        let env = CartPole().wrapped { env in
            env.orderEnforced().timeLimited(100)
        }
        
        #expect(env is TimeLimit<OrderEnforcing<CartPole>>)
    }
    
    @Test("Observation transform applies correctly")
    func testObservationTransformApplies() {
        var env = CartPole().observationsTransformed { obs in
            obs * 2
        }
        
        let (obs, _) = env.reset(seed: 42, options: nil)
        
        var baseEnv = CartPole()
        let (baseObs, _) = baseEnv.reset(seed: 42, options: nil)
        
        let obsValues: [Float] = obs.asArray(Float.self)
        let baseValues: [Float] = baseObs.asArray(Float.self)
        
        for i in 0..<obsValues.count {
            #expect(abs(obsValues[i] - baseValues[i] * 2) < 1e-5)
        }
    }
    
    @Test("Normalized observations have reasonable values")
    func testNormalizedObservationsReasonable() {
        var env = CartPole().observationsNormalized()
        
        _ = env.reset(seed: 42, options: nil)
        
        for _ in 0..<100 {
            let result = env.step(Int.random(in: 0..<2))
            let values: [Float] = result.obs.asArray(Float.self)
            
            for value in values {
                #expect(abs(value) < 100, "Normalized values should be bounded")
            }
            
            if result.terminated || result.truncated {
                _ = env.reset(seed: nil, options: nil)
            }
        }
    }
}

