//
//  AsyncVectorEnvTests.swift
//

import Testing
import MLX
@testable import Gymnazo

@MainActor
@Suite("Async Vector Environment Tests")
struct AsyncVectorEnvTests {
    
    @Test("AsyncVectorEnv initialization")
    func testAsyncVectorEnvInitialization() {
        let envs = AsyncVectorEnv(envFns: [
            { CartPole() },
            { CartPole() },
            { CartPole() }
        ])
        
        #expect(envs.num_envs == 3)
        #expect(envs.closed == false)
        #expect(envs.autoreset_mode == .nextStep)
    }
    
    @Test("AsyncVectorEnv from factory functions")
    func testAsyncVectorEnvFromFactoryFunctions() {
        let envs = AsyncVectorEnv(envFns: [
            { CartPole() },
            { CartPole() }
        ])
        
        #expect(envs.num_envs == 2)
    }
    
    @Test("Reset returns correct shape")
    func testResetReturnsCorrectShape() {
        let envs = AsyncVectorEnv(envFns: [
            { CartPole() },
            { CartPole() },
            { CartPole() }
        ])
        
        let result = envs.reset(seed: 42)
        
        #expect(result.observations.shape == [3, 4])
    }
    
    @Test("Async reset returns correct shape")
    func testAsyncResetReturnsCorrectShape() async {
        let envs = AsyncVectorEnv(envFns: [
            { CartPole() },
            { CartPole() },
            { CartPole() },
            { CartPole() }
        ])
        
        let result = await envs.resetAsync(seed: 42)
        
        #expect(result.observations.shape == [4, 4])
    }
    
    @Test("Reset with seed produces deterministic results")
    func testResetSeedDeterminism() {
        let envs1 = AsyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        let envs2 = AsyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        
        let result1 = envs1.reset(seed: 123)
        let result2 = envs2.reset(seed: 123)
        
        let obs1Values: [Float] = result1.observations.asArray(Float.self)
        let obs2Values: [Float] = result2.observations.asArray(Float.self)
        
        for i in 0..<obs1Values.count {
            #expect(abs(obs1Values[i] - obs2Values[i]) < 1e-5)
        }
    }
    
    @Test("Seed is incremented per environment")
    func testResetSeedIsIncrementedPerEnv() {
        let envs = AsyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        
        let result = envs.reset(seed: 100)
        
        let obs: [[Float]] = (0..<2).map { i in
            let slice = result.observations[i]
            return slice.asArray(Float.self)
        }
        
        #expect(obs[0] != obs[1])
    }
    
    @Test("Step returns correct shapes")
    func testStepReturnsCorrectShapes() {
        let envs = AsyncVectorEnv(envFns: [
            { CartPole() },
            { CartPole() },
            { CartPole() }
        ])
        
        _ = envs.reset(seed: 42)
        let result = envs.step([0, 1, 0])
        
        #expect(result.observations.shape == [3, 4])
        #expect(result.rewards.shape == [3])
        #expect(result.terminations.shape == [3])
        #expect(result.truncations.shape == [3])
    }
    
    @Test("Async step returns correct shapes")
    func testAsyncStepReturnsCorrectShapes() async {
        let envs = AsyncVectorEnv(envFns: [
            { CartPole() },
            { CartPole() },
            { CartPole() },
            { CartPole() }
        ])
        
        _ = await envs.resetAsync(seed: 42)
        let result = await envs.stepAsync([0, 1, 0, 1])
        
        #expect(result.observations.shape == [4, 4])
        #expect(result.rewards.shape == [4])
        #expect(result.terminations.shape == [4])
        #expect(result.truncations.shape == [4])
    }
    
    @Test("Step with discrete actions")
    func testStepWithDiscreteActions() {
        let envs = AsyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        
        _ = envs.reset(seed: 42)
        let result = envs.step([1, 0])
        
        let rewardsArray: [Float] = result.rewards.asArray(Float.self)
        #expect(rewardsArray.count == 2)
        #expect(rewardsArray[0] == 1.0)
        #expect(rewardsArray[1] == 1.0)
    }
    
    @Test("Multiple steps work correctly")
    func testMultipleSteps() {
        let envs = AsyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        
        _ = envs.reset(seed: 42)
        
        for _ in 0..<10 {
            let result = envs.step([0, 1])
            #expect(result.observations.shape == [2, 4])
        }
    }
    
    @Test("Multiple async steps work correctly")
    func testMultipleAsyncSteps() async {
        let envs = AsyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        
        _ = await envs.resetAsync(seed: 42)
        
        for _ in 0..<10 {
            let result = await envs.stepAsync([0, 1])
            #expect(result.observations.shape == [2, 4])
        }
    }
    
    @Test("Autoreset stores final observation")
    func testAutoResetStoresFinalObservation() {
        let envs = AsyncVectorEnv(envFns: [{ CartPole() }])
        
        _ = envs.reset(seed: 42)
        
        var foundFinalObs = false
        for _ in 0..<1000 {
            let result = envs.step([1])
            
            if let finals = result.finals, finals.observations[0] != nil {
                foundFinalObs = true
                break
            }
        }
        
        #expect(foundFinalObs, "Should have found final_observation in autoreset")
    }

    @Test("Same-step autoreset returns reset observation")
    func testSameStepAutoresetReturnsResetObservation() async {
        let envs = AsyncVectorEnv(envFns: [{ CartPole() }], autoresetMode: .sameStep)
        
        _ = await envs.resetAsync(seed: 42)
        
        var terminated = false
        var checked = false
        
        for _ in 0..<500 {
            let result = await envs.stepAsync([0])
            eval(result.terminations)
            let term = result.terminations[0].item(Bool.self)
            
            if term {
                terminated = true
                #expect(result.finals?.indices.contains(0) == true)
                #expect(result.finals?.observations[0] != nil)
                
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
    func testAutoResetIndicesTracked() {
        let envs = AsyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        
        _ = envs.reset(seed: 42)
        
        for _ in 0..<1000 {
            let result = envs.step([1, 1])
            
            if let indices = result.finals?.indices {
                #expect(!indices.isEmpty)
                break
            }
        }
    }
    
    @Test("Close environment")
    func testClose() {
        let envs = AsyncVectorEnv(envFns: [{ CartPole() }])
        
        _ = envs.reset(seed: 42)
        envs.close()
        
        #expect(envs.closed == true)
    }
    
    @Test("Double close is idempotent")
    func testDoubleCloseIsIdempotent() {
        let envs = AsyncVectorEnv(envFns: [{ CartPole() }])
        
        _ = envs.reset(seed: 42)
        envs.close()
        envs.close()
        
        #expect(envs.closed == true)
    }
    
    @Test("Single observation space")
    func testSingleObservationSpace() {
        let envs = AsyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        
        #expect(envs.single_observation_space.shape == [4])
    }
    
    @Test("Batched observation space")
    func testBatchedObservationSpace() {
        let envs = AsyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }, { CartPole() }])
        
        #expect(envs.observation_space.shape == [3, 4])
    }
    
    @Test("Single action space")
    func testSingleActionSpace() {
        let envs = AsyncVectorEnv(envFns: [{ CartPole() }])
        
        if let discrete = envs.single_action_space as? Discrete {
            #expect(discrete.n == 2)
        } else {
            Issue.record("Expected Discrete action space")
        }
    }
    
    @Test("Batched action space is MultiDiscrete")
    func testBatchedActionSpaceIsMultiDiscrete() {
        let envs = AsyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        
        if let multiDiscrete = envs.action_space as? MultiDiscrete {
            #expect(multiDiscrete.shape == [2])
        } else {
            Issue.record("Expected MultiDiscrete action space")
        }
    }
    
    @Test("Single environment works")
    func testSingleEnvironment() {
        let envs = AsyncVectorEnv(envFns: [{ CartPole() }])
        
        let resetResult = envs.reset(seed: 42)
        #expect(resetResult.observations.shape == [1, 4])
        
        let stepResult = envs.step([0])
        #expect(stepResult.observations.shape == [1, 4])
    }
    
    @Test("Many environments work")
    func testManyEnvironments() {
        let envs = AsyncVectorEnv(envFns: (0..<10).map { _ in { CartPole() } })
        
        #expect(envs.num_envs == 10)
        
        let resetResult = envs.reset(seed: 42)
        #expect(resetResult.observations.shape == [10, 4])
        
        let stepResult = envs.step(Array(repeating: 0, count: 10))
        #expect(stepResult.observations.shape == [10, 4])
    }
    
    @Test("make_vec_async function works")
    func testMakeVecAsyncFunction() {
        let envs = make_vec_async("CartPole", numEnvs: 3)
        
        #expect(envs.num_envs == 3)
        
        let result = envs.reset(seed: 42)
        #expect(result.observations.shape == [3, 4])
    }
    
    @Test("make_vec_async with envFns works")
    func testMakeVecAsyncWithEnvFns() {
        let envs = make_vec_async(envFns: [
            { CartPole() },
            { CartPole() }
        ])
        
        #expect(envs.num_envs == 2)
    }
    
    @Test("make_vec with async mode returns AsyncVectorEnv")
    func testMakeVecWithAsyncMode() {
        let envs = make_vec("CartPole", numEnvs: 2, vectorizationMode: .async)
        
        #expect(envs is AsyncVectorEnv)
        #expect(envs.num_envs == 2)
    }
    
    @Test("make_vec with sync mode returns SyncVectorEnv")
    func testMakeVecWithSyncMode() {
        let envs = make_vec("CartPole", numEnvs: 2, vectorizationMode: .sync)
        
        #expect(envs is SyncVectorEnv)
        #expect(envs.num_envs == 2)
    }
    
    @Test("Async and sync produce same results with same seed")
    func testAsyncAndSyncProduceSameResults() {
        let syncEnvs = SyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        let asyncEnvs = AsyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        
        let syncResult = syncEnvs.reset(seed: 42)
        let asyncResult = asyncEnvs.reset(seed: 42)
        
        let syncObs: [Float] = syncResult.observations.asArray(Float.self)
        let asyncObs: [Float] = asyncResult.observations.asArray(Float.self)
        
        for i in 0..<syncObs.count {
            #expect(abs(syncObs[i] - asyncObs[i]) < 1e-5)
        }
    }
    
    @Test("Different seeds produce different results")
    func testDifferentSeedsProduceDifferentResults() {
        let envs1 = AsyncVectorEnv(envFns: [{ CartPole() }])
        let envs2 = AsyncVectorEnv(envFns: [{ CartPole() }])
        
        let result1 = envs1.reset(seed: 1)
        let result2 = envs2.reset(seed: 2)
        
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

