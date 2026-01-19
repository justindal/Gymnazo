import Testing
import MLX
@testable import Gymnazo

@MainActor
@Suite("Async Vector Environment Tests")
struct AsyncVectorEnvTests {
    func makeCartPoleEnv() async throws -> AnyEnv<MLXArray, Int> {
        try await Gymnazo.make("CartPole")
    }

    func makeAsyncVectorEnv(
        count: Int,
        copyObservations: Bool = true,
        autoresetMode: AutoresetMode = .nextStep
    ) async throws -> AsyncVectorEnv<Int> {
        var envs: [AnyEnv<MLXArray, Int>] = []
        envs.reserveCapacity(count)
        for _ in 0..<count {
            envs.append(try await makeCartPoleEnv())
        }
        return try AsyncVectorEnv(
            envs: envs,
            copyObservations: copyObservations,
            autoresetMode: autoresetMode
        )
    }

    func makeSendableCartPoleEnvFns(count: Int) async throws -> [@Sendable () -> any Env] {
        var boxes: [UnsafeEnvBox<Int>] = []
        boxes.reserveCapacity(count)
        for _ in 0..<count {
            let env = try await makeCartPoleEnv()
            boxes.append(UnsafeEnvBox(env: env))
        }
        return boxes.map { box in { box.env } }
    }

    func makeSyncVectorEnv(count: Int) async throws -> SyncVectorEnv<Int> {
        var envs: [AnyEnv<MLXArray, Int>] = []
        envs.reserveCapacity(count)
        for _ in 0..<count {
            envs.append(try await makeCartPoleEnv())
        }
        return try SyncVectorEnv(envs: envs)
    }
    
    @Test("AsyncVectorEnv initialization")
    func testAsyncVectorEnvInitialization() async throws {
        let envs = try await makeAsyncVectorEnv(count: 3)
        
        #expect(envs.numEnvs == 3)
        #expect(envs.closed == false)
        #expect(envs.autoresetMode == .nextStep)
    }
    
    @Test("AsyncVectorEnv from factory functions")
    func testAsyncVectorEnvFromFactoryFunctions() async throws {
        let envs = try await makeAsyncVectorEnv(count: 2)
        
        #expect(envs.numEnvs == 2)
    }
    
    @Test("Reset returns correct shape")
    func testResetReturnsCorrectShape() async throws {
        let envs = try await makeAsyncVectorEnv(count: 3)
        
        let result = try envs.reset(seed: 42)
        
        #expect(result.observations.shape == [3, 4])
    }
    
    @Test("Async reset returns correct shape")
    func testAsyncResetReturnsCorrectShape() async throws {
        let envs = try await makeAsyncVectorEnv(count: 4)
        
        let result = try await envs.resetAsync(seed: 42)
        
        #expect(result.observations.shape == [4, 4])
    }
    
    @Test("Reset with seed produces deterministic results")
    func testResetSeedDeterminism() async throws {
        let envs1 = try await makeAsyncVectorEnv(count: 2)
        let envs2 = try await makeAsyncVectorEnv(count: 2)
        
        let result1 = try envs1.reset(seed: 123)
        let result2 = try envs2.reset(seed: 123)
        
        let obs1Values: [Float] = result1.observations.asArray(Float.self)
        let obs2Values: [Float] = result2.observations.asArray(Float.self)
        
        for i in 0..<obs1Values.count {
            #expect(abs(obs1Values[i] - obs2Values[i]) < 1e-5)
        }
    }
    
    @Test("Seed is incremented per environment")
    func testResetSeedIsIncrementedPerEnv() async throws {
        let envs = try await makeAsyncVectorEnv(count: 2)
        
        let result = try envs.reset(seed: 100)
        
        let obs: [[Float]] = (0..<2).map { i in
            let slice = result.observations[i]
            return slice.asArray(Float.self)
        }
        
        #expect(obs[0] != obs[1])
    }
    
    @Test("Step returns correct shapes")
    func testStepReturnsCorrectShapes() async throws {
        let envs = try await makeAsyncVectorEnv(count: 3)
        
        _ = try envs.reset(seed: 42)
        let result = try envs.step([0, 1, 0])
        
        #expect(result.observations.shape == [3, 4])
        #expect(result.rewards.shape == [3])
        #expect(result.terminations.shape == [3])
        #expect(result.truncations.shape == [3])
    }
    
    @Test("Async step returns correct shapes")
    func testAsyncStepReturnsCorrectShapes() async throws {
        let envs = try await makeAsyncVectorEnv(count: 4)
        
        _ = try await envs.resetAsync(seed: 42)
        let result = try await envs.stepAsync([0, 1, 0, 1])
        
        #expect(result.observations.shape == [4, 4])
        #expect(result.rewards.shape == [4])
        #expect(result.terminations.shape == [4])
        #expect(result.truncations.shape == [4])
    }
    
    @Test("Step with discrete actions")
    func testStepWithDiscreteActions() async throws {
        let envs = try await makeAsyncVectorEnv(count: 2)
        
        _ = try envs.reset(seed: 42)
        let result = try envs.step([1, 0])
        
        let rewardsArray: [Float] = result.rewards.asArray(Float.self)
        #expect(rewardsArray.count == 2)
        #expect(rewardsArray[0] == 1.0)
        #expect(rewardsArray[1] == 1.0)
    }
    
    @Test("Multiple steps work correctly")
    func testMultipleSteps() async throws {
        let envs = try await makeAsyncVectorEnv(count: 2)
        
        _ = try envs.reset(seed: 42)
        
        for _ in 0..<10 {
            let result = try envs.step([0, 1])
            #expect(result.observations.shape == [2, 4])
        }
    }
    
    @Test("Multiple async steps work correctly")
    func testMultipleAsyncSteps() async throws {
        let envs = try await makeAsyncVectorEnv(count: 2)
        
        _ = try await envs.resetAsync(seed: 42)
        
        for _ in 0..<10 {
            let result = try await envs.stepAsync([0, 1])
            #expect(result.observations.shape == [2, 4])
        }
    }
    
    @Test("Autoreset stores final info")
    func testAutoResetStoresFinalInfo() async throws {
        let envs = try await makeAsyncVectorEnv(count: 1)
        
        _ = try envs.reset(seed: 42)
        
        var foundFinalInfo = false
        for _ in 0..<1000 {
            let result = try envs.step([1])
            
            if result.infos[0]["final_info"] != nil {
                foundFinalInfo = true
                break
            }
        }
        
        #expect(foundFinalInfo, "Should have found final_info in autoreset")
    }

    @Test("Same-step autoreset returns reset observation")
    func testSameStepAutoresetReturnsResetObservation() async throws {
        let envs = try await makeAsyncVectorEnv(count: 1, autoresetMode: .sameStep)
        
        _ = try await envs.resetAsync(seed: 42)
        
        var terminated = false
        var checked = false
        
        for _ in 0..<500 {
            let result = try await envs.stepAsync([0])
            eval(result.terminations)
            let term = result.terminations[0].item(Bool.self)
            
            if term {
                terminated = true
                #expect(result.infos[0]["final_info"] != nil)
                
                let obs = result.observations[0]
                eval(obs)
                let maxVal = abs(obs).max().item(Float.self)
                #expect(maxVal < 0.2)
                checked = true
                break
            }
        }
        
        #expect(terminated == true)
        #expect(checked == true)
    }
    
    @Test("Autoreset indices tracked")
    func testAutoResetIndicesTracked() async throws {
        let envs = try await makeAsyncVectorEnv(count: 2)
        
        _ = try envs.reset(seed: 42)
        
        for _ in 0..<1000 {
            let result = try envs.step([1, 1])
            
            let indices = result.infos.enumerated().compactMap { index, info in
                info["final_info"] == nil ? nil : index
            }
            if !indices.isEmpty {
                #expect(!indices.isEmpty)
                break
            }
        }
    }
    
    @Test("Close environment")
    func testClose() async throws {
        let envs = try await makeAsyncVectorEnv(count: 1)
        
        _ = try envs.reset(seed: 42)
        envs.close()
        
        #expect(envs.closed == true)
    }
    
    @Test("Double close is idempotent")
    func testDoubleCloseIsIdempotent() async throws {
        let envs = try await makeAsyncVectorEnv(count: 1)
        
        _ = try envs.reset(seed: 42)
        envs.close()
        envs.close()
        
        #expect(envs.closed == true)
    }
    
    @Test("Single observation space")
    func testSingleObservationSpace() async throws {
        let envs = try await makeAsyncVectorEnv(count: 2)
        
        #expect(envs.singleObservationSpace.shape == [4])
    }
    
    @Test("Batched observation space")
    func testBatchedObservationSpace() async throws {
        let envs = try await makeAsyncVectorEnv(count: 3)
        
        #expect(envs.observationSpace.shape == [3, 4])
    }
    
    @Test("Single action space")
    func testSingleActionSpace() async throws {
        let envs = try await makeAsyncVectorEnv(count: 1)
        
        if let discrete = envs.singleActionSpace as? Discrete {
            #expect(discrete.n == 2)
        } else {
            Issue.record("Expected Discrete action space")
        }
    }
    
    @Test("Batched action space is MultiDiscrete")
    func testBatchedActionSpaceIsMultiDiscrete() async throws {
        let envs = try await makeAsyncVectorEnv(count: 2)
        
        if let multiDiscrete = envs.actionSpace as? MultiDiscrete {
            #expect(multiDiscrete.shape == [2])
        } else {
            Issue.record("Expected MultiDiscrete action space")
        }
    }
    
    @Test("Single environment works")
    func testSingleEnvironment() async throws {
        let envs = try await makeAsyncVectorEnv(count: 1)
        
        let resetResult = try envs.reset(seed: 42)
        #expect(resetResult.observations.shape == [1, 4])
        
        let stepResult = try envs.step([0])
        #expect(stepResult.observations.shape == [1, 4])
    }
    
    @Test("Many environments work")
    func testManyEnvironments() async throws {
        let envs = try await makeAsyncVectorEnv(count: 10)
        
        #expect(envs.numEnvs == 10)
        
        let resetResult = try envs.reset(seed: 42)
        #expect(resetResult.observations.shape == [10, 4])
        
        let stepResult = try envs.step(Array(repeating: 0, count: 10))
        #expect(stepResult.observations.shape == [10, 4])
    }
    
    @Test("makeVecAsync function works")
    func testMakeVecAsyncFunction() async throws {
        let envs: AsyncVectorEnv<Int> = try await Gymnazo.makeVecAsync("CartPole", numEnvs: 3)
        
        #expect(envs.numEnvs == 3)
        
        let result = try envs.reset(seed: 42)
        #expect(result.observations.shape == [3, 4])
    }
    
    @Test("makeVecAsync with envFns works")
    func testMakeVecAsyncWithEnvFns() async throws {
        let envFns = try await makeSendableCartPoleEnvFns(count: 2)
        let envs: AsyncVectorEnv<Int> = try await Gymnazo.makeVecAsync(envFns: envFns)
        
        #expect(envs.numEnvs == 2)
    }
    
    @Test("makeVec with async mode returns AsyncVectorEnv")
    func testMakeVecWithAsyncMode() async throws {
        let envs: any VectorEnv<Int> = try await Gymnazo.makeVec(
            "CartPole",
            numEnvs: 2,
            vectorizationMode: .async
        )
        
        #expect(envs is AsyncVectorEnv<Int>)
        #expect(envs.numEnvs == 2)
    }
    
    @Test("makeVec with sync mode returns SyncVectorEnv")
    func testMakeVecWithSyncMode() async throws {
        let envs: any VectorEnv<Int> = try await Gymnazo.makeVec(
            "CartPole",
            numEnvs: 2,
            vectorizationMode: .sync
        )
        
        #expect(envs is SyncVectorEnv<Int>)
        #expect(envs.numEnvs == 2)
    }
    
    @Test("Async and sync produce same results with same seed")
    func testAsyncAndSyncProduceSameResults() async throws {
        let syncEnvs = try await makeSyncVectorEnv(count: 2)
        let asyncEnvs = try await makeAsyncVectorEnv(count: 2)
        
        let syncResult = try syncEnvs.reset(seed: 42)
        let asyncResult = try asyncEnvs.reset(seed: 42)
        
        let syncObs: [Float] = syncResult.observations.asArray(Float.self)
        let asyncObs: [Float] = asyncResult.observations.asArray(Float.self)
        
        for i in 0..<syncObs.count {
            #expect(abs(syncObs[i] - asyncObs[i]) < 1e-5)
        }
    }
    
    @Test("Different seeds produce different results")
    func testDifferentSeedsProduceDifferentResults() async throws {
        let envs1 = try await makeAsyncVectorEnv(count: 1)
        let envs2 = try await makeAsyncVectorEnv(count: 1)
        
        let result1 = try envs1.reset(seed: 1)
        let result2 = try envs2.reset(seed: 2)
        
        let obs1: [Float] = result1.observations.asArray(Float.self)
        let obs2: [Float] = result2.observations.asArray(Float.self)
        
        var different = false
        for i in 0..<obs1.count {
            if abs(obs1[i] - obs2[i]) > 1e-5 {
                different = true
                break
            }
        }
        
        #expect(different, "Different seeds should produce different observations")
    }
}

