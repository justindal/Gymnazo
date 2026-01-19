import Testing
import Foundation
import MLX
@testable import Gymnazo

@Suite("Acrobot environment")
struct AcrobotTests {
    func makeAcrobot(renderMode: RenderMode? = nil) async throws -> Acrobot {
        let options: EnvOptions = renderMode.map { ["render_mode": $0.rawValue] } ?? [:]
        let env: AnyEnv<MLXArray, Int> = try await Gymnazo.make("Acrobot", options: options)
        guard let acrobot = env.unwrapped as? Acrobot else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "Acrobot",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return acrobot
    }
    
    @Test
    func testInitialization() async throws {
        let env = try await makeAcrobot()
        
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
        let env = try await makeAcrobot()

        guard let actionSpace = env.actionSpace as? Discrete else {
            Issue.record("Action space is not Discrete")
            return
        }

        #expect(actionSpace.n == 3)
        #expect(actionSpace.start == 0)
    }
    
    @Test
    func testObservationSpace() async throws {
        let env = try await makeAcrobot()
        
        #expect(env.observationSpace.shape == [6])
    }
    
    @Test
    func testObservationSpaceBounds() async throws {
        let env = try await makeAcrobot()

        guard let observationSpace = env.observationSpace as? Box else {
            Issue.record("Observation space is not Box")
            return
        }
        
        let low = observationSpace.low
        let high = observationSpace.high
        
        eval(low, high)
        
        for i in 0..<4 {
            #expect(low[i].item(Float.self) == -1.0)
            #expect(high[i].item(Float.self) == 1.0)
        }
        
        #expect(low[4].item(Float.self) == -4 * Float.pi)
        #expect(high[4].item(Float.self) == 4 * Float.pi)
        #expect(low[5].item(Float.self) == -9 * Float.pi)
        #expect(high[5].item(Float.self) == 9 * Float.pi)
    }
    
    @Test
    func testRenderModeInitialization() async throws {
        let envNoRender = try await makeAcrobot(renderMode: nil)
        #expect(envNoRender.renderMode == nil)
        
        let envHuman = try await makeAcrobot(renderMode: .human)
        #expect(envHuman.renderMode == .human)
    }
    
    @Test
    func testResetReturnsObservation() async throws {
        var env = try await makeAcrobot()
        let result = try env.reset(seed: 42)
        let obs = result.obs
        let info = result.info
        
        eval(obs)
        
        #expect(obs.shape == [6])
        #expect(info.isEmpty || info.count >= 0)
    }
    
    @Test
    func testResetDeterminismWithSeed() async throws {
        var env1 = try await makeAcrobot()
        var env2 = try await makeAcrobot()
        
        let obs1 = try env1.reset(seed: 123).obs
        let obs2 = try env2.reset(seed: 123).obs
        
        eval(obs1, obs2)
        
        let diff = abs(obs1 - obs2).sum().item(Float.self)
        #expect(diff < 1e-6)
    }
    
    @Test
    func testResetDifferentSeedsGiveDifferentStates() async throws {
        var env1 = try await makeAcrobot()
        var env2 = try await makeAcrobot()
        
        let obs1 = try env1.reset(seed: 1).obs
        let obs2 = try env2.reset(seed: 999).obs
        
        eval(obs1, obs2)
        
        let diff = abs(obs1 - obs2).sum().item(Float.self)
        #expect(diff > 1e-6)
    }
    
    @Test
    func testResetStateWithinBounds() async throws {
        var env = try await makeAcrobot()
        
        for seed in 0..<10 {
            _ = try env.reset(seed: UInt64(seed))
            
            guard let state = env.state else {
                Issue.record("State should not be nil after reset")
                return
            }
            
            for i in 0..<4 {
                #expect(state[i] >= -0.1 && state[i] <= 0.1)
            }
        }
    }
    
    @Test
    func testResetWithCustomBounds() async throws {
        var env = try await makeAcrobot()
        _ = try env.reset(seed: 42, options: ["low": Float(-0.2), "high": Float(0.2)])
        
        guard let state = env.state else {
            Issue.record("State should not be nil after reset")
            return
        }
        
        for i in 0..<4 {
            #expect(state[i] >= -0.2 && state[i] <= 0.2)
        }
    }
    
    @Test
    func testStepReturnsValidResult() async throws {
        var env = try await makeAcrobot()
        _ = try env.reset(seed: 42)
        
        let result = try env.step(1)
        
        eval(result.obs)
        
        #expect(result.obs.shape == [6])
        #expect(result.reward == -1.0 || result.reward == 0.0)
        #expect(result.truncated == false)
    }
    
    @Test
    func testStepRewardIsMinusOneWhileNotTerminated() async throws {
        var env = try await makeAcrobot()
        _ = try env.reset(seed: 42)
        
        for _ in 0..<10 {
            let result = try env.step(1)
            if !result.terminated {
                #expect(result.reward == -1.0)
            } else {
                #expect(result.reward == 0.0)
            }
        }
    }
    
    @Test
    func testObservationIsValidTrigonometric() async throws {
        var env = try await makeAcrobot()
        _ = try env.reset(seed: 42)
        
        for _ in 0..<50 {
            let result = try env.step(Int.random(in: 0..<3))
            eval(result.obs)
            
            let cosTheta1 = result.obs[0].item(Float.self)
            let sinTheta1 = result.obs[1].item(Float.self)
            let cosTheta2 = result.obs[2].item(Float.self)
            let sinTheta2 = result.obs[3].item(Float.self)
            
            #expect(cosTheta1 >= -1.0 && cosTheta1 <= 1.0)
            #expect(sinTheta1 >= -1.0 && sinTheta1 <= 1.0)
            #expect(cosTheta2 >= -1.0 && cosTheta2 <= 1.0)
            #expect(sinTheta2 >= -1.0 && sinTheta2 <= 1.0)
            
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
        var env = try await makeAcrobot()
        _ = try env.reset(seed: 42)
        
        for _ in 0..<100 {
            let result = try env.step(Int.random(in: 0..<3))
            
            guard let state = env.state else {
                continue
            }
            
            #expect(state[2] >= -4 * Float.pi && state[2] <= 4 * Float.pi)
            #expect(state[3] >= -9 * Float.pi && state[3] <= 9 * Float.pi)
            
            if result.terminated {
                break
            }
        }
    }
    
    @Test
    func testTerminationCondition() async throws {
        var env = try await makeAcrobot()
        _ = try env.reset(seed: 42)
        
        var terminated = false
        for _ in 0..<500 {
            let action = Int.random(in: 0..<3)
            let result = try env.step(action)
            
            if result.terminated {
                terminated = true
                
                guard let state = env.state else { break }
                let theta1 = state[0]
                let theta2 = state[1]
                let height = -Foundation.cos(theta1) - Foundation.cos(theta1 + theta2)
                #expect(height > 1.0)
                
                #expect(result.reward == 0.0)
                break
            }
        }
        
        _ = terminated
    }
    
    @Test
    func testDeterministicStepSequence() async throws {
        var env1 = try await makeAcrobot()
        var env2 = try await makeAcrobot()
        
        _ = try env1.reset(seed: 777)
        _ = try env2.reset(seed: 777)
        
        let actions = [0, 1, 2, 0, 1, 2, 1, 0]
        
        for action in actions {
            let result1 = try env1.step(action)
            let result2 = try env2.step(action)
            
            eval(result1.obs, result2.obs)
            
            let diff = abs(result1.obs - result2.obs).sum().item(Float.self)
            #expect(diff < 1e-5)
            #expect(result1.reward == result2.reward)
            #expect(result1.terminated == result2.terminated)
        }
    }
    
    @Test
    func testAvailableTorques() async throws {
        let env = try await makeAcrobot()
        
        #expect(env.availableTorques.count == 3)
        #expect(env.availableTorques[0] == -1.0)
        #expect(env.availableTorques[1] == 0.0)
        #expect(env.availableTorques[2] == 1.0)
    }
    
    @Test
    func testMaxVelocities() async throws {
        let env = try await makeAcrobot()
        
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
        let snapshotDown = AcrobotSnapshot(theta1: 0, theta2: 0, linkLength1: 1.0, linkLength2: 1.0)
        
        #expect(abs(snapshotDown.p1.x - 0.0) < 0.001)
        #expect(abs(snapshotDown.p1.y - (-1.0)) < 0.001)
        
        #expect(abs(snapshotDown.p2.x - 0.0) < 0.001)
        #expect(abs(snapshotDown.p2.y - (-2.0)) < 0.001)
    }
    
    @Test
    func testCurrentSnapshotBeforeReset() async throws {
        let env = try await makeAcrobot()
        let snapshot = env.currentSnapshot
        
        #expect(snapshot == AcrobotSnapshot.zero)
    }
    
    @Test
    func testCurrentSnapshotAfterReset() async throws {
        var env = try await makeAcrobot()
        _ = try env.reset(seed: 42)
        
        let snapshot = env.currentSnapshot
        
        guard let state = env.state else {
            Issue.record("State should not be nil")
            return
        }
        
        #expect(snapshot.theta1 == state[0])
        #expect(snapshot.theta2 == state[1])
    }
    
    @Test
    func testCurrentSnapshotUpdatesAfterStep() async throws {
        var env = try await makeAcrobot()
        _ = try env.reset(seed: 42)
        
        let snapshotBefore = env.currentSnapshot
        _ = try env.step(2)
        let snapshotAfter = env.currentSnapshot
        
        #expect(snapshotBefore != snapshotAfter)
    }
    
    @Test
    func testRenderReturnsNilWhenNoRenderMode() async throws {
        var env = try await makeAcrobot(renderMode: nil)
        _ = try env.reset(seed: 42)
        
        let result = try env.render()
        #expect(result == nil)
    }
    
    @Test
    func testRenderReturnsNilForUnsupportedMode() async throws {
        var env = try await makeAcrobot(renderMode: .ansi)
        _ = try env.reset(seed: 42)
        
        let result = try env.render()
        #expect(result == nil)
    }
    
    @Test
    func testRenderRgbArrayNotImplemented() async throws {
        var env = try await makeAcrobot(renderMode: .rgbArray)
        _ = try env.reset(seed: 42)
        
        let result = try env.render()
        #expect(result == nil)
    }
    
    @Test
    func testBookOrNipsDefault() async throws {
        let env = try await makeAcrobot()
        #expect(env.bookOrNips == "book")
    }
}

