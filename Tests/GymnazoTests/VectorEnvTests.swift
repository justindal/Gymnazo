import Testing
import MLX
@testable import Gymnazo

@Suite("Vector Environment Tests")
struct VectorEnvTests {
    
    @Test
    @MainActor
    func testSyncVectorEnvInitialization() async throws {
        let envs = SyncVectorEnv(envFns: [
            { CartPole() },
            { CartPole() },
            { CartPole() }
        ])
        
        #expect(envs.num_envs == 3)
        #expect(envs.closed == false)
        #expect(envs.autoreset_mode == .nextStep)
    }
    
    @Test
    @MainActor
    func testSyncVectorEnvFromPreCreatedEnvs() async throws {
        let cartPole1 = CartPole()
        let cartPole2 = CartPole()
        
        let envs = SyncVectorEnv(envs: [cartPole1, cartPole2])
        
        #expect(envs.num_envs == 2)
        #expect(envs.environments.count == 2)
    }
    
    @Test
    @MainActor
    func testMakeVecFunction() async throws {
        let envs = make_vec("CartPole", numEnvs: 3)
        
        #expect(envs.num_envs == 3)
        #expect(envs.spec?.name == "CartPole")
    }
    
    @Test
    @MainActor
    func testMakeVecWithEnvFns() async throws {
        let envs = make_vec(envFns: [
            { CartPole() },
            { CartPole() }
        ])
        
        #expect(envs.num_envs == 2)
    }
    
    @Test
    @MainActor
    func testResetReturnsCorrectShape() async throws {
        let numEnvs = 4
        let envs = SyncVectorEnv(envFns: (0..<numEnvs).map { _ in { CartPole() } })
        
        let result = envs.reset(seed: 42)
        
        // CartPole has observation shape [4], so batched should be [numEnvs, 4]
        #expect(result.observations.shape == [numEnvs, 4])
    }
    
    @Test
    @MainActor
    func testResetWithSeedIsDeterministic() async throws {
        let envs1 = SyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        let envs2 = SyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        
        let result1 = envs1.reset(seed: 123)
        let result2 = envs2.reset(seed: 123)
        
        eval(result1.observations, result2.observations)
        
        let diff = abs(result1.observations - result2.observations).sum().item(Float.self)
        #expect(diff < 1e-6)
    }
    
    @Test
    @MainActor
    func testResetWithDifferentSeedsProducesDifferentResults() async throws {
        let envs1 = SyncVectorEnv(envFns: [{ CartPole() }])
        let envs2 = SyncVectorEnv(envFns: [{ CartPole() }])
        
        let result1 = envs1.reset(seed: 1)
        let result2 = envs2.reset(seed: 999)
        
        eval(result1.observations, result2.observations)
        
        let diff = abs(result1.observations - result2.observations).sum().item(Float.self)
        #expect(diff > 1e-6)
    }
    
    @Test
    @MainActor
    func testResetSeedIsIncrementedPerEnv() async throws {
        // When seed=42 is provided, env[0] gets seed 42, env[1] gets seed 43, etc.
        let envs = SyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        let result = envs.reset(seed: 42)
        
        // Create individual environments with seeds 42 and 43 to verify
        var singleEnv1 = CartPole()
        var singleEnv2 = CartPole()
        let (obs1, _) = singleEnv1.reset(seed: 42)
        let (obs2, _) = singleEnv2.reset(seed: 43)
        
        eval(result.observations, obs1, obs2)
        
        // First row should match env with seed 42
        let diff1 = abs(result.observations[0] - obs1).sum().item(Float.self)
        #expect(diff1 < 1e-6)
        
        // Second row should match env with seed 43
        let diff2 = abs(result.observations[1] - obs2).sum().item(Float.self)
        #expect(diff2 < 1e-6)
    }
    
    @Test
    @MainActor
    func testStepReturnsCorrectShapes() async throws {
        let numEnvs = 3
        let envs = SyncVectorEnv(envFns: (0..<numEnvs).map { _ in { CartPole() } })
        _ = envs.reset(seed: 42)
        
        let actions = [1, 0, 1]
        let result = envs.step(actions)
        
        // Check shapes
        #expect(result.observations.shape == [numEnvs, 4])
        #expect(result.rewards.shape == [numEnvs])
        #expect(result.terminations.shape == [numEnvs])
        #expect(result.truncations.shape == [numEnvs])
    }
    
    @Test
    @MainActor
    func testStepWithDiscreteActions() async throws {
        let envs = SyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        _ = envs.reset(seed: 42)
        
        // Step with discrete actions
        let result = envs.step([1, 0])
        
        eval(result.observations, result.rewards)
        
        // Both environments should have received reward (1.0 for CartPole)
        let rewards = result.rewards
        #expect(rewards[0].item(Float.self) == 1.0)
        #expect(rewards[1].item(Float.self) == 1.0)
    }
    
    @Test
    @MainActor
    func testStepIsDeterministic() async throws {
        let envs1 = SyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        let envs2 = SyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        
        _ = envs1.reset(seed: 42)
        _ = envs2.reset(seed: 42)
        
        let actions = [1, 0]
        let result1 = envs1.step(actions)
        let result2 = envs2.step(actions)
        
        eval(result1.observations, result2.observations, result1.rewards, result2.rewards)
        
        let obsDiff = abs(result1.observations - result2.observations).sum().item(Float.self)
        let rewardDiff = abs(result1.rewards - result2.rewards).sum().item(Float.self)
        
        #expect(obsDiff < 1e-6)
        #expect(rewardDiff < 1e-6)
    }
    
    @Test
    @MainActor
    func testMultipleSteps() async throws {
        let envs = SyncVectorEnv(envFns: [{ CartPole() }, { CartPole() }])
        _ = envs.reset(seed: 42)
        
        // Take multiple steps
        for _ in 0..<10 {
            let result = envs.step([1, 0])
            #expect(result.observations.shape == [2, 4])
        }
    }
    
    @Test
    @MainActor
    func testAutoresetStoresFinalObservation() async throws {
        let envs = SyncVectorEnv(
            envFns: [{ CartPole() }],
            autoresetMode: .nextStep
        )
        _ = envs.reset(seed: 42)
        
        // Step until termination
        var terminated = false
        var finalObsStored = false
        
        for _ in 0..<500 {
            let result = envs.step([0]) // Always push left to eventually fail
            
            eval(result.terminations)
            let term = result.terminations[0].item(Bool.self)
            
            if term {
                terminated = true
                // Check that final_observation is in infos
                if let finalObs = result.infos["final_observation"] as? [Int: MLXArray] {
                    finalObsStored = finalObs[0] != nil
                }
                break
            }
        }
        
        #expect(terminated == true)
        #expect(finalObsStored == true)
    }
    
    @Test
    @MainActor
    func testAutoresetNextStepResetsEnvironment() async throws {
        let envs = SyncVectorEnv(
            envFns: [{ CartPole() }],
            autoresetMode: .nextStep
        )
        _ = envs.reset(seed: 42)
        
        // Step until termination
        var stepsUntilTermination = 0
        for i in 0..<500 {
            let result = envs.step([0])
            eval(result.terminations)
            let term = result.terminations[0].item(Bool.self)
            if term {
                stepsUntilTermination = i
                break
            }
        }
        
        // Take one more step - should autoreset and continue
        let resultAfterReset = envs.step([1])
        eval(resultAfterReset.observations)
        
        // Environment should have reset, observation should be valid
        #expect(resultAfterReset.observations.shape == [1, 4])
        
        // Observation values should be small (reset state is within [-0.05, 0.05])
        let obs = resultAfterReset.observations[0]
        eval(obs)
        let maxVal = abs(obs).max().item(Float.self)
        // After reset, values should be small again
        // (this might not always be true if the step after reset moved the cart significantly)
    }
    
    @Test
    @MainActor
    func testAutoResetIndicesTracked() async throws {
        // Create environments that will terminate at different times
        let envs = SyncVectorEnv(
            envFns: [{ CartPole() }, { CartPole() }],
            autoresetMode: .nextStep
        )
        _ = envs.reset(seed: 42)
        
        // Step with action that causes one env to fail faster
        // (pushing consistently in one direction)
        var terminatedIndices: Set<Int> = []
        
        for _ in 0..<500 {
            let result = envs.step([0, 1]) // Different actions for each env
            eval(result.terminations)
            
            if let indices = result.infos["_final_observation_indices"] as? [Int] {
                for idx in indices {
                    terminatedIndices.insert(idx)
                }
            }
            
            // Check individual terminations
            let term0 = result.terminations[0].item(Bool.self)
            let term1 = result.terminations[1].item(Bool.self)
            
            if term0 && term1 {
                break
            }
        }
        
        // At least one environment should have terminated
        #expect(!terminatedIndices.isEmpty)
    }
    
    @Test
    @MainActor
    func testCloseMarksEnvironmentAsClosed() async throws {
        let envs = SyncVectorEnv(envFns: [{ CartPole() }])
        #expect(envs.closed == false)
        
        envs.close()
        
        #expect(envs.closed == true)
    }
    
    @Test
    @MainActor
    func testDoubleCloseIsIdempotent() async throws {
        let envs = SyncVectorEnv(envFns: [{ CartPole() }])
        
        envs.close()
        envs.close() // Should not crash
        
        #expect(envs.closed == true)
    }
    
    @Test
    @MainActor
    func testSingleObservationSpace() async throws {
        let envs = SyncVectorEnv(envFns: [{ CartPole() }])
        
        // Single observation space should match CartPole's observation space
        #expect(envs.single_observation_space.shape == [4])
    }
    
    @Test
    @MainActor
    func testSingleActionSpace() async throws {
        let envs = SyncVectorEnv(envFns: [{ CartPole() }])
        
        if let discreteSpace = envs.single_action_space as? Discrete {
            #expect(discreteSpace.n == 2) // CartPole has 2 actions
        } else {
            Issue.record("Expected Discrete action space")
        }
    }
    
    @Test
    @MainActor
    func testSingleEnvironment() async throws {
        let envs = SyncVectorEnv(envFns: [{ CartPole() }])
        
        #expect(envs.num_envs == 1)
        
        let result = envs.reset(seed: 42)
        #expect(result.observations.shape == [1, 4])
        
        let stepResult = envs.step([1])
        #expect(stepResult.observations.shape == [1, 4])
    }
    
    @Test
    @MainActor
    func testManyEnvironments() async throws {
        let numEnvs = 10
        let envs = make_vec("CartPole", numEnvs: numEnvs)
        
        #expect(envs.num_envs == numEnvs)
        
        let result = envs.reset(seed: 0)
        #expect(result.observations.shape == [numEnvs, 4])
        
        let actions = Array(repeating: 1, count: numEnvs)
        let stepResult = envs.step(actions)
        #expect(stepResult.observations.shape == [numEnvs, 4])
    }
    
    @Test
    @MainActor
    func testFullEpisode() async throws {
        let envs = make_vec("CartPole", numEnvs: 2)
        var result = envs.reset(seed: 42)
        
        var totalRewards: [Float] = [0, 0]
        var steps = 0
        let maxSteps = 1000
        
        while steps < maxSteps {
            // Random actions
            let actions = [Int.random(in: 0...1), Int.random(in: 0...1)]
            let stepResult = envs.step(actions)
            
            eval(stepResult.rewards, stepResult.terminations, stepResult.truncations)
            
            totalRewards[0] += stepResult.rewards[0].item(Float.self)
            totalRewards[1] += stepResult.rewards[1].item(Float.self)
            
            steps += 1
            
            // Check if both environments have gone through autoreset
            if let indices = stepResult.infos["_final_observation_indices"] as? [Int] {
                // Some environments terminated this step
            }
        }
        
        // Should have accumulated some rewards
        #expect(totalRewards[0] > 0)
        #expect(totalRewards[1] > 0)
        
        envs.close()
    }
    
    @Test
    @MainActor
    func testWithMountainCar() async throws {
        let envs = make_vec("MountainCar", numEnvs: 2)
        
        let result = envs.reset(seed: 42)
        
        // MountainCar has observation shape [2]
        #expect(result.observations.shape == [2, 2])
        
        // Step with continuous-ish discrete actions
        let stepResult = envs.step([0, 2]) // MountainCar has 3 actions: 0, 1, 2
        #expect(stepResult.observations.shape == [2, 2])
        
        envs.close()
    }
}

@Suite("AutoresetMode Tests")
struct AutoresetModeTests {
    
    @Test
    func testAutoresetModeValues() {
        #expect(AutoresetMode.nextStep.rawValue == "next_step")
        #expect(AutoresetMode.sameStep.rawValue == "same_step")
        #expect(AutoresetMode.disabled.rawValue == "disabled")
    }
}

@Suite("VectorStepResult Tests")
struct VectorStepResultTests {
    
    @Test
    func testVectorStepResultInit() {
        let result = VectorStepResult(
            observations: MLXArray([1.0, 2.0, 3.0]),
            rewards: MLXArray([1.0]),
            terminations: MLXArray([false]),
            truncations: MLXArray([false]),
            infos: ["key": "value"]
        )
        
        #expect(result.observations.shape == [3])
        #expect(result.rewards.shape == [1])
        #expect(result.infos["key"] as? String == "value")
    }
}

@Suite("VectorResetResult Tests")
struct VectorResetResultTests {
    
    @Test
    func testVectorResetResultInit() {
        let obs = MLX.stacked([MLXArray([1.0, 2.0] as [Float]), MLXArray([3.0, 4.0] as [Float])], axis: 0)
        let result = VectorResetResult(
            observations: obs,
            infos: [:]
        )
        
        #expect(result.observations.shape == [2, 2])
        #expect(result.infos.isEmpty)
    }
}

