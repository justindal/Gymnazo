import Testing
import MLX
@testable import Gymnazo

@Suite("Vector Environment Tests")
struct VectorEnvTests {
    @MainActor
    func makeCartPoleEnv() async throws -> AnyEnv<MLXArray, Int> {
        try await Gymnazo.make("CartPole")
    }

    @MainActor
    func makeCartPoleEnvFns(count: Int) async throws -> [() -> any Env] {
        var envs: [AnyEnv<MLXArray, Int>] = []
        envs.reserveCapacity(count)
        for _ in 0..<count {
            envs.append(try await makeCartPoleEnv())
        }
        return envs.map { env in { env } }
    }

    @MainActor
    func makeSyncVectorEnv(
        count: Int,
        copyObservations: Bool = true,
        autoresetMode: AutoresetMode = .nextStep
    ) async throws -> SyncVectorEnv<Int> {
        let envFns = try await makeCartPoleEnvFns(count: count)
        return try SyncVectorEnv(
            envFns: envFns,
            copyObservations: copyObservations,
            autoresetMode: autoresetMode
        )
    }
    
    @Test
    @MainActor
    func testSyncVectorEnvInitialization() async throws {
        let envs = try await makeSyncVectorEnv(count: 3)
        
        #expect(envs.numEnvs == 3)
        #expect(envs.closed == false)
        #expect(envs.autoresetMode == .nextStep)
    }
    
    @Test
    @MainActor
    func testSyncVectorEnvFromEnvFns() async throws {
        let envFns = try await makeCartPoleEnvFns(count: 2)
        let envs: SyncVectorEnv<Int> = try SyncVectorEnv(envFns: envFns)
        
        #expect(envs.numEnvs == 2)
    }
    
    @Test
    @MainActor
    func testMakeVecFunction() async throws {
        let envs: SyncVectorEnv<Int> = try await Gymnazo.makeVec("CartPole", numEnvs: 3)
        
        #expect(envs.numEnvs == 3)
        #expect(envs.spec?.name == "CartPole")
    }
    
    @Test
    @MainActor
    func testMakeVecWithEnvFns() async throws {
        let envFns = try await makeCartPoleEnvFns(count: 2)
        let envs: SyncVectorEnv<Int> = try await Gymnazo.makeVec(envFns: envFns)
        
        #expect(envs.numEnvs == 2)
    }
    
    @Test
    @MainActor
    func testResetReturnsCorrectShape() async throws {
        let numEnvs = 4
        let envs = try await makeSyncVectorEnv(count: numEnvs)
        
        let result = try envs.reset(seed: 42)
        
        #expect(result.observations.shape == [numEnvs, 4])
    }
    
    @Test
    @MainActor
    func testResetWithSeedIsDeterministic() async throws {
        let envs1 = try await makeSyncVectorEnv(count: 2)
        let envs2 = try await makeSyncVectorEnv(count: 2)
        
        let result1 = try envs1.reset(seed: 123)
        let result2 = try envs2.reset(seed: 123)
        
        eval(result1.observations, result2.observations)
        
        let diff = abs(result1.observations - result2.observations).sum().item(Float.self)
        #expect(diff < 1e-6)
    }
    
    @Test
    @MainActor
    func testResetWithDifferentSeedsProducesDifferentResults() async throws {
        let envs1 = try await makeSyncVectorEnv(count: 1)
        let envs2 = try await makeSyncVectorEnv(count: 1)
        
        let result1 = try envs1.reset(seed: 1)
        let result2 = try envs2.reset(seed: 999)
        
        eval(result1.observations, result2.observations)
        
        let diff = abs(result1.observations - result2.observations).sum().item(Float.self)
        #expect(diff > 1e-6)
    }
    
    @Test
    @MainActor
    func testResetSeedIsIncrementedPerEnv() async throws {
        let envs = try await makeSyncVectorEnv(count: 2)
        let result = try envs.reset(seed: 42)
        
        var singleEnv1 = try await makeCartPoleEnv()
        var singleEnv2 = try await makeCartPoleEnv()
        let obs1 = try singleEnv1.reset(seed: 42).obs
        let obs2 = try singleEnv2.reset(seed: 43).obs
        
        eval(result.observations, obs1, obs2)
        
        let diff1 = abs(result.observations[0] - obs1).sum().item(Float.self)
        #expect(diff1 < 1e-6)
        
        let diff2 = abs(result.observations[1] - obs2).sum().item(Float.self)
        #expect(diff2 < 1e-6)
    }
    
    @Test
    @MainActor
    func testStepReturnsCorrectShapes() async throws {
        let numEnvs = 3
        let envs = try await makeSyncVectorEnv(count: numEnvs)
        _ = try envs.reset(seed: 42)
        
        let actions = [1, 0, 1]
        let result = try envs.step(actions)
        
        #expect(result.observations.shape == [numEnvs, 4])
        #expect(result.rewards.shape == [numEnvs])
        #expect(result.terminations.shape == [numEnvs])
        #expect(result.truncations.shape == [numEnvs])
    }
    
    @Test
    @MainActor
    func testStepWithDiscreteActions() async throws {
        let envs = try await makeSyncVectorEnv(count: 2)
        _ = try envs.reset(seed: 42)
        
        let result = try envs.step([1, 0])
        
        eval(result.observations, result.rewards)
        
        let rewards = result.rewards
        #expect(rewards[0].item(Float.self) == 1.0)
        #expect(rewards[1].item(Float.self) == 1.0)
    }
    
    @Test
    @MainActor
    func testStepIsDeterministic() async throws {
        let envs1 = try await makeSyncVectorEnv(count: 2)
        let envs2 = try await makeSyncVectorEnv(count: 2)
        
        _ = try envs1.reset(seed: 42)
        _ = try envs2.reset(seed: 42)
        
        let actions = [1, 0]
        let result1 = try envs1.step(actions)
        let result2 = try envs2.step(actions)
        
        eval(result1.observations, result2.observations, result1.rewards, result2.rewards)
        
        let obsDiff = abs(result1.observations - result2.observations).sum().item(Float.self)
        let rewardDiff = abs(result1.rewards - result2.rewards).sum().item(Float.self)
        
        #expect(obsDiff < 1e-6)
        #expect(rewardDiff < 1e-6)
    }
    
    @Test
    @MainActor
    func testMultipleSteps() async throws {
        let envs = try await makeSyncVectorEnv(count: 2)
        _ = try envs.reset(seed: 42)
        
        for _ in 0..<10 {
            let result = try envs.step([1, 0])
            #expect(result.observations.shape == [2, 4])
        }
    }
    
    @Test
    @MainActor
    func testAutoresetStoresFinalInfo() async throws {
        let envs = try await makeSyncVectorEnv(count: 1, autoresetMode: .nextStep)
        _ = try envs.reset(seed: 42)
        
        var terminated = false
        var finalInfoStored = false
        
        for _ in 0..<500 {
            let result = try envs.step([0])
            
            eval(result.terminations)
            let term = result.terminations[0].item(Bool.self)
            
            if term {
                terminated = true
                if result.infos[0]["final_info"] != nil {
                    finalInfoStored = true
                }
                break
            }
        }
        
        #expect(terminated == true)
        #expect(finalInfoStored == true)
    }

    @Test
    @MainActor
    func testSameStepAutoresetReturnsResetObservation() async throws {
        let envs = try await makeSyncVectorEnv(count: 1, autoresetMode: .sameStep)
        _ = try envs.reset(seed: 42)
        
        var terminated = false
        var checked = false
        
        for _ in 0..<500 {
            let result = try envs.step([0])
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
    
    @Test
    @MainActor
    func testAutoresetNextStepResetsEnvironment() async throws {
        let envs = try await makeSyncVectorEnv(count: 1, autoresetMode: .nextStep)
        _ = try envs.reset(seed: 42)
        
        var stepsUntilTermination = 0
        for i in 0..<500 {
            let result = try envs.step([0])
            eval(result.terminations)
            let term = result.terminations[0].item(Bool.self)
            if term {
                stepsUntilTermination = i
                break
            }
        }
        
        let resultAfterReset = try envs.step([1])
        eval(resultAfterReset.observations)
        
        #expect(resultAfterReset.observations.shape == [1, 4])
        
        let obs = resultAfterReset.observations[0]
        eval(obs)
        let maxVal = abs(obs).max().item(Float.self)
    }
    
    @Test
    @MainActor
    func testAutoResetIndicesTracked() async throws {
        let envs = try await makeSyncVectorEnv(count: 2, autoresetMode: .nextStep)
        _ = try envs.reset(seed: 42)
        
        var terminatedIndices: Set<Int> = []
        
        for _ in 0..<500 {
            let result = try envs.step([0, 1])
            eval(result.terminations)
            
            for (idx, info) in result.infos.enumerated() {
                if info["final_info"] != nil {
                    terminatedIndices.insert(idx)
                }
            }
            
            let term0 = result.terminations[0].item(Bool.self)
            let term1 = result.terminations[1].item(Bool.self)
            
            if term0 && term1 {
                break
            }
        }
        
        #expect(!terminatedIndices.isEmpty)
    }
    
    @Test
    @MainActor
    func testCloseMarksEnvironmentAsClosed() async throws {
        let envs = try await makeSyncVectorEnv(count: 1)
        #expect(envs.closed == false)
        
        envs.close()
        
        #expect(envs.closed == true)
    }
    
    @Test
    @MainActor
    func testDoubleCloseIsIdempotent() async throws {
        let envs = try await makeSyncVectorEnv(count: 1)
        
        envs.close()
        envs.close()
        
        #expect(envs.closed == true)
    }
    
    @Test
    @MainActor
    func testSingleObservationSpace() async throws {
        let envs = try await makeSyncVectorEnv(count: 1)
        
        #expect(envs.singleObservationSpace.shape == [4])
    }
    
    @Test
    @MainActor
    func testSingleActionSpace() async throws {
        let envs = try await makeSyncVectorEnv(count: 1)
        
        if let discreteSpace = envs.singleActionSpace as? Discrete {
            #expect(discreteSpace.n == 2)
        } else {
            Issue.record("Expected Discrete action space")
        }
    }
    
    @Test
    @MainActor
    func testSingleEnvironment() async throws {
        let envs = try await makeSyncVectorEnv(count: 1)
        
        #expect(envs.numEnvs == 1)
        
        let result = try envs.reset(seed: 42)
        #expect(result.observations.shape == [1, 4])
        
        let stepResult = try envs.step([1])
        #expect(stepResult.observations.shape == [1, 4])
    }
    
    @Test
    @MainActor
    func testManyEnvironments() async throws {
        let numEnvs = 10
        let envs: SyncVectorEnv<Int> = try await Gymnazo.makeVec("CartPole", numEnvs: numEnvs)
        
        #expect(envs.numEnvs == numEnvs)
        
        let result = try envs.reset(seed: 0)
        #expect(result.observations.shape == [numEnvs, 4])
        
        let actions = Array(repeating: 1, count: numEnvs)
        let stepResult = try envs.step(actions)
        #expect(stepResult.observations.shape == [numEnvs, 4])
    }
    
    @Test
    @MainActor
    func testFullEpisode() async throws {
        let envs: SyncVectorEnv<Int> = try await Gymnazo.makeVec("CartPole", numEnvs: 2)
        var result = try envs.reset(seed: 42)
        
        var totalRewards: [Float] = [0, 0]
        var steps = 0
        let maxSteps = 1000
        
        while steps < maxSteps {
            let actions = [Int.random(in: 0...1), Int.random(in: 0...1)]
            let stepResult = try envs.step(actions)
            
            eval(stepResult.rewards, stepResult.terminations, stepResult.truncations)
            
            totalRewards[0] += stepResult.rewards[0].item(Float.self)
            totalRewards[1] += stepResult.rewards[1].item(Float.self)
            
            steps += 1
            
        }
        
        #expect(totalRewards[0] > 0)
        #expect(totalRewards[1] > 0)
        
        envs.close()
    }
    
    @Test
    @MainActor
    func testWithMountainCar() async throws {
        let envs: SyncVectorEnv<Int> = try await Gymnazo.makeVec("MountainCar", numEnvs: 2)
        
        let result = try envs.reset(seed: 42)
        
        #expect(result.observations.shape == [2, 2])
        
        let stepResult = try envs.step([0, 2])
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
        let info: Info = ["key": "value"]
        let result = VectorStepResult(
            observations: MLXArray([1.0, 2.0, 3.0]),
            rewards: MLXArray([1.0]),
            terminations: MLXArray([false]),
            truncations: MLXArray([false]),
            infos: [info]
        )
        
        #expect(result.observations.shape == [3])
        #expect(result.rewards.shape == [1])
        #expect(result.infos[0]["key"]?.string == "value")
    }
}

@Suite("VectorResetResult Tests")
struct VectorResetResultTests {
    
    @Test
    func testVectorResetResultInit() {
        let obs = MLX.stacked([MLXArray([1.0, 2.0] as [Float]), MLXArray([3.0, 4.0] as [Float])], axis: 0)
        let result = VectorResetResult(
            observations: obs,
            infos: [Info(), Info()]
        )
        
        #expect(result.observations.shape == [2, 2])
        #expect(result.infos.count == 2)
        #expect(result.infos[0].isEmpty)
        #expect(result.infos[1].isEmpty)
    }
}

