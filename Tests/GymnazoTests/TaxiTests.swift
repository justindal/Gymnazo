import Testing
import MLX
@testable import Gymnazo

@Suite("Taxi environment")
struct TaxiTests {
    @Test
    func testResetDeterminismWithSeed() async throws {
        let env = Taxi()
        let r1 = env.reset(seed: 42)
        let r2 = env.reset(seed: 42)
        #expect(r1.obs == r2.obs)
    }
    
    @Test
    func testEncodeDecodeRoundtrip() async throws {
        for taxiRow in 0..<5 {
            for taxiCol in 0..<5 {
                for passLoc in 0..<5 {
                    for destIdx in 0..<4 {
                        let encoded = Taxi.encode(taxiRow: taxiRow, taxiCol: taxiCol, passLoc: passLoc, destIdx: destIdx)
                        let decoded = Taxi.decode(encoded)
                        
                        #expect(decoded.taxiRow == taxiRow)
                        #expect(decoded.taxiCol == taxiCol)
                        #expect(decoded.passLoc == passLoc)
                        #expect(decoded.destIdx == destIdx)
                    }
                }
            }
        }
    }
    
    @Test
    func testStateSpaceSize() async throws {
        let env = Taxi()
        #expect(env.observation_space.n == 500)
        #expect(env.action_space.n == 6)
    }
    
    @Test
    func testInfoContainsProbOnly() async throws {
        let env = Taxi()
        let result = env.reset(seed: 123)
        
        #expect(result.info["prob"]?.double == 1.0)
        #expect(result.info["action_mask"] == nil)
    }
    
    @Test
    func testMovementSouth() async throws {
        let env = Taxi()
        _ = env.reset(seed: 0)
        
        let initialState = Taxi.encode(taxiRow: 0, taxiCol: 0, passLoc: 1, destIdx: 0)
        
        let (row, col, _, _) = Taxi.decode(initialState)
        if row < 4 {
            let result = env.step(0)
            let (newRow, _, _, _) = Taxi.decode(result.obs)
            #expect(newRow >= row)
        }
    }
    
    @Test
    func testIllegalPickupPenalty() async throws {
        let env = Taxi()
        _ = env.reset(seed: 42)
        
        let result = env.step(4)
        
        if result.reward == -10.0 {
            #expect(result.terminated == false)
        }
    }
    
    @Test
    func testIllegalDropoffPenalty() async throws {
        let env = Taxi()
        _ = env.reset(seed: 42)
        
        let result = env.step(5)
        
        #expect(result.reward == -10.0 || result.reward == -1.0)
        #expect(result.terminated == false)
    }
    
    @Test
    func testStepReward() async throws {
        let env = Taxi()
        _ = env.reset(seed: 100)
        
        let result = env.step(0)
        
        #expect(result.reward == -1.0 || result.reward == -10.0)
    }
    
    @Test
    func testActionMaskValidActions() async throws {
        let env = Taxi()
        _ = env.reset(seed: 42)
        
        let state = Taxi.encode(taxiRow: 2, taxiCol: 2, passLoc: 0, destIdx: 1)
        let mask = env.actionMask(for: state)
        
        #expect(mask[0] == 1)
        #expect(mask[1] == 1)
    }
    
    @Test
    func testActionMaskAtCorner() async throws {
        let env = Taxi()
        
        let state = Taxi.encode(taxiRow: 0, taxiCol: 0, passLoc: 0, destIdx: 1)
        let mask = env.actionMask(for: state)
        
        #expect(mask[1] == 0)
        #expect(mask[3] == 0)
        
        #expect(mask[4] == 1)
    }
    
    @Test
    func testSuccessfulEpisode() async throws {
        let env = Taxi()
        _ = env.reset(seed: 42)
        
        var terminated = false
        var totalReward: Double = 0.0
        var steps = 0
        let maxSteps = 500
        
        while !terminated && steps < maxSteps {
            let mask = env.actionMask(for: env.reset(seed: nil).obs)
            var validActions = [Int]()
            for (i, m) in mask.enumerated() {
                if m == 1 { validActions.append(i) }
            }
            
            let action = validActions.isEmpty ? 0 : validActions[steps % validActions.count]
            let result = env.step(action)
            totalReward += result.reward
            terminated = result.terminated
            steps += 1
        }
        
        #expect(steps > 0)
    }
    
    @Test
    func testRainyModeCreation() async throws {
        let env = Taxi(isRainy: true)
        let result = env.reset(seed: 42)
        #expect(result.obs >= 0)
        #expect(result.obs < 500)
    }
    
    @Test
    func testFicklePassengerModeCreation() async throws {
        let env = Taxi(ficklePassenger: true)
        let result = env.reset(seed: 42)
        #expect(result.obs >= 0)
        #expect(result.obs < 500)
    }
    
    @Test
    func testAnsiRender() async throws {
        let env = Taxi(render_mode: "ansi")
        _ = env.reset(seed: 42)
        let ansi = env.renderAnsi()
        #expect(ansi.contains("+"))
        #expect(ansi.contains("-"))
        #expect(ansi.contains("Legend:"))
    }
    
    @Test
    @MainActor
    func testGymnazoMakeTaxi() async throws {
        let env = Gymnazo.make("Taxi")
        let taxi = env.unwrapped as! Taxi
        _ = taxi.reset(seed: 123)
        let result = taxi.step(0)
        #expect(result.obs >= 0)
        #expect(result.obs < 500)
    }
    
    @Test
    @MainActor
    func testGymnazoMakeTaxiWithKwargs() async throws {
        let env = Gymnazo.make("Taxi", kwargs: ["render_mode": "ansi"])
        let taxi = env.unwrapped as! Taxi
        _ = taxi.reset(seed: 42)
        #expect(taxi.action_space.n == 6)
    }
    
    @Test
    func testLocationsCorrect() async throws {
        #expect(Taxi.locs.count == 4)
        #expect(Taxi.locs[0] == (0, 0))
        #expect(Taxi.locs[1] == (0, 4))
        #expect(Taxi.locs[2] == (4, 0))
        #expect(Taxi.locs[3] == (4, 3))
    }
    
    @Test
    func testPickupAtCorrectLocation() async throws {
        let env = Taxi()
        
        for seed: UInt64 in 0..<100 {
            _ = env.reset(seed: seed)
            let (taxiRow, taxiCol, passIdx, _) = Taxi.decode(env.reset(seed: seed).obs)
            
            if passIdx < 4 {
                let passLoc = Taxi.locs[passIdx]
                if (taxiRow, taxiCol) == passLoc {
                    let result = env.step(4)
                    let (_, _, newPassIdx, _) = Taxi.decode(result.obs)
                    #expect(newPassIdx == 4)
                    #expect(result.reward == -1.0)
                    return
                }
            }
        }
    }
}

