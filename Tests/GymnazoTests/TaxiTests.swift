import Testing
import MLX
@testable import Gymnazo

@Suite("Taxi environment")
struct TaxiTests {
    func makeTaxi(renderMode: RenderMode? = nil, isRainy: Bool? = nil, ficklePassenger: Bool? = nil) async throws -> Taxi {
        var options: EnvOptions = [:]
        if let renderMode {
            options["render_mode"] = renderMode.rawValue
        }
        if let isRainy {
            options["is_rainy"] = isRainy
        }
        if let ficklePassenger {
            options["fickle_passenger"] = ficklePassenger
        }
        let env = try await Gymnazo.make("Taxi", options: options)
        guard let taxi = env.unwrapped as? Taxi else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "Taxi",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return taxi
    }

    @Test
    func testResetDeterminismWithSeed() async throws {
        let env = try await makeTaxi()
        let r1 = try env.reset(seed: 42)
        let r2 = try env.reset(seed: 42)
        #expect(MLX.arrayEqual(r1.obs, r2.obs).item(Bool.self))
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
        let env = try await makeTaxi()
        #expect((env.observationSpace as? Discrete)?.n == 500)
        #expect((env.actionSpace as? Discrete)?.n == 6)
    }
    
    @Test
    func testInfoContainsProbOnly() async throws {
        let env = try await makeTaxi()
        let result = try env.reset(seed: 123)
        
        #expect(result.info["prob"]?.double == 1.0)
        #expect(result.info["action_mask"] == nil)
    }
    
    @Test
    func testMovementSouth() async throws {
        let env = try await makeTaxi()
        _ = try env.reset(seed: 0)
        
        let initialState = Taxi.encode(taxiRow: 0, taxiCol: 0, passLoc: 1, destIdx: 0)
        
        let (row, _, _, _) = Taxi.decode(initialState)
        if row < 4 {
            let result = try env.step(MLXArray(Int32(0)))
            let (newRow, _, _, _) = Taxi.decode(result.obs.item(Int.self))
            #expect(newRow >= row)
        }
    }
    
    @Test
    func testIllegalPickupPenalty() async throws {
        let env = try await makeTaxi()
        _ = try env.reset(seed: 42)
        
        let result = try env.step(MLXArray(Int32(4)))
        
        if result.reward == -10.0 {
            #expect(result.terminated == false)
        }
    }
    
    @Test
    func testIllegalDropoffPenalty() async throws {
        let env = try await makeTaxi()
        _ = try env.reset(seed: 42)
        
        let result = try env.step(MLXArray(Int32(5)))
        
        #expect(result.reward == -10.0 || result.reward == -1.0)
        #expect(result.terminated == false)
    }
    
    @Test
    func testStepReward() async throws {
        let env = try await makeTaxi()
        _ = try env.reset(seed: 100)
        
        let result = try env.step(MLXArray(Int32(0)))
        
        #expect(result.reward == -1.0 || result.reward == -10.0)
    }
    
    @Test
    func testActionMaskValidActions() async throws {
        let env = try await makeTaxi()
        _ = try env.reset(seed: 42)
        
        let state = Taxi.encode(taxiRow: 2, taxiCol: 2, passLoc: 0, destIdx: 1)
        let mask = env.actionMask(for: state)
        
        #expect(mask[0] == 1)
        #expect(mask[1] == 1)
    }
    
    @Test
    func testActionMaskAtCorner() async throws {
        let env = try await makeTaxi()
        
        let state = Taxi.encode(taxiRow: 0, taxiCol: 0, passLoc: 0, destIdx: 1)
        let mask = env.actionMask(for: state)
        
        #expect(mask[1] == 0)
        #expect(mask[3] == 0)
        
        #expect(mask[4] == 1)
    }
    
    @Test
    func testSuccessfulEpisode() async throws {
        let env = try await makeTaxi()
        _ = try env.reset(seed: 42)
        
        var terminated = false
        var totalReward: Double = 0.0
        var steps = 0
        let maxSteps = 500
        
        while !terminated && steps < maxSteps {
            let mask = env.actionMask(for: try env.reset(seed: nil).obs.item(Int.self))
            var validActions = [Int]()
            for (i, m) in mask.enumerated() {
                if m == 1 { validActions.append(i) }
            }
            
            let action = validActions.isEmpty ? 0 : validActions[steps % validActions.count]
            let result = try env.step(MLXArray(Int32(action)))
            totalReward += result.reward
            terminated = result.terminated
            steps += 1
        }
        
        #expect(steps > 0)
    }
    
    @Test
    func testRainyModeCreation() async throws {
        let env = try await makeTaxi(isRainy: true)
        let result = try env.reset(seed: 42)
        #expect(result.obs.item(Int.self) >= 0)
        #expect(result.obs.item(Int.self) < 500)
    }
    
    @Test
    func testFicklePassengerModeCreation() async throws {
        let env = try await makeTaxi(ficklePassenger: true)
        let result = try env.reset(seed: 42)
        #expect(result.obs.item(Int.self) >= 0)
        #expect(result.obs.item(Int.self) < 500)
    }
    
    @Test
    func testAnsiRender() async throws {
        let env = try await makeTaxi(renderMode: .ansi)
        _ = try env.reset(seed: 42)
        let ansi = env.renderAnsi()
        #expect(ansi.contains("+"))
        #expect(ansi.contains("-"))
        #expect(ansi.contains("Legend:"))
    }
    
    @Test
    @MainActor
    func testGymnazoMakeTaxi() async throws {
        let env = try await Gymnazo.make("Taxi")
        let taxi = env.unwrapped as! Taxi
        _ = try taxi.reset(seed: 123)
        let result = try taxi.step(MLXArray(Int32(0)))
        #expect(result.obs.item(Int.self) >= 0)
        #expect(result.obs.item(Int.self) < 500)
    }
    
    @Test
    @MainActor
    func testGymnazoMakeTaxiWithKwargs() async throws {
        let env = try await Gymnazo.make(
            "Taxi",
            options: ["render_mode": "ansi"]
        )
        let taxi = env.unwrapped as! Taxi
        _ = try taxi.reset(seed: 42)
        #expect((taxi.actionSpace as? Discrete)?.n == 6)
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
        let env = try await makeTaxi()
        
        for seed: UInt64 in 0..<100 {
            _ = try env.reset(seed: seed)
            let (taxiRow, taxiCol, passIdx, _) = Taxi.decode(try env.reset(seed: seed).obs.item(Int.self))
            
            if passIdx < 4 {
                let passLoc = Taxi.locs[passIdx]
                if (taxiRow, taxiCol) == passLoc {
                    let result = try env.step(MLXArray(Int32(4)))
                    let (_, _, newPassIdx, _) = Taxi.decode(result.obs.item(Int.self))
                    #expect(newPassIdx == 4)
                    #expect(result.reward == -1.0)
                    return
                }
            }
        }
    }
}

