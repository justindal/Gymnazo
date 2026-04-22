import Testing
import Foundation
import MLX
import MLXNN
@testable import Gymnazo

@Suite("Persistence Tests", .serialized)
struct PersistenceTests {
    private func tempDir() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("gymnazo_tests_\(UUID().uuidString)")
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    @Test @MainActor
    func tabularAgentQLearningSaveLoad() async throws {
        let dir = tempDir()
        defer { cleanup(dir) }

        let env = try await Gymnazo.make("FrozenLake")
        let config = TabularConfig(learningRate: 0.3, gamma: 0.9)
        let agent = TabularAgent(
            updateRule: .qLearning,
            config: config,
            numStates: 16,
            numActions: 4,
            seed: 42,
            qTable: MLX.ones([16, 4]) * 0.5,
            timesteps: 500,
            explorationRate: 0.25
        )
        agent.setEnv(env)

        try await agent.save(to: dir)

        let loaded = try TabularAgent.load(from: dir, env: env)

        #expect(await loaded.numTimesteps == 500)
        #expect(await loaded.epsilon == 0.25)
        #expect(await loaded.numStates == 16)
        #expect(await loaded.numActions == 4)
        let loadedConfig = await loaded.config
        #expect(loadedConfig.learningRate == 0.3)
        #expect(loadedConfig.gamma == 0.9)
        #expect(await loaded.updateRule == .qLearning)

        let originalQ = await agent.tableValues()
        let loadedQ = await loaded.tableValues()
        #expect(originalQ.count == loadedQ.count)
        for i in 0..<originalQ.count {
            #expect(abs(originalQ[i] - loadedQ[i]) < 1e-6)
        }
    }

    @Test @MainActor
    func tabularAgentSarsaSaveLoad() async throws {
        let dir = tempDir()
        defer { cleanup(dir) }

        let env = try await Gymnazo.make("FrozenLake")
        let agent = TabularAgent(
            updateRule: .sarsa,
            config: TabularConfig(),
            numStates: 16,
            numActions: 4,
            seed: 42,
            qTable: MLX.ones([16, 4]) * 0.3,
            timesteps: 200,
            explorationRate: 0.5
        )
        agent.setEnv(env)

        try await agent.save(to: dir)

        let loaded = try TabularAgent.load(from: dir, env: env)

        #expect(await loaded.numTimesteps == 200)
        #expect(await loaded.epsilon == 0.5)
        #expect(await loaded.updateRule == .sarsa)

        let originalQ = await agent.tableValues()
        let loadedQ = await loaded.tableValues()
        #expect(originalQ.count == loadedQ.count)
        for i in 0..<originalQ.count {
            #expect(abs(originalQ[i] - loadedQ[i]) < 1e-6)
        }
    }

    @Test @MainActor
    func dqnSaveLoad() async throws {
        let dir = tempDir()
        defer { cleanup(dir) }

        let env = try await Gymnazo.make("CartPole")
        let dqnConfig = DQNConfig(
            bufferSize: 1000,
            learningStarts: 10,
            batchSize: 32
        )
        let dqn = try DQN(
            observationSpace: env.observationSpace,
            actionSpace: env.actionSpace as! Discrete,
            config: dqnConfig,
            seed: 42
        )
        dqn.setEnv(env)

        await dqn.restore(
            timesteps: 100,
            totalTimesteps: 500,
            progressRemaining: 0.8,
            explorationRate: 0.7,
            gradientSteps: 50
        )

        try await dqn.save(to: dir)

        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("metadata.json").path))
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("policy.safetensors").path))
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("target.safetensors").path))

        let loaded = try DQN.load(from: dir, env: env)

        #expect(await loaded.numTimesteps == 100)
        #expect(await loaded.totalTimesteps == 500)
        #expect(abs(await loaded.progressRemaining - 0.8) < 1e-6)
        #expect(await loaded.gradientSteps == 50)
        #expect(abs(await loaded.explorationRate - 0.7) < 1e-6)
        let loadedDqnConfig = loaded.config
        #expect(loadedDqnConfig.bufferSize == 1000)
        #expect(loadedDqnConfig.batchSize == 32)
    }

    @Test @MainActor
    func sacSaveLoad() async throws {
        let dir = tempDir()
        defer { cleanup(dir) }

        let env = try await Gymnazo.make("Pendulum")
        let config = OffPolicyConfig(
            bufferSize: 500,
            learningStarts: 10,
            batchSize: 32
        )
        let networksConfig = SACNetworksConfig(
            actor: SACActorConfig(
                netArch: .shared([128, 64]),
                featuresExtractor: .flatten
            ),
            critic: SACCriticConfig(
                netArch: [128, 64],
                nCritics: 2,
                shareFeaturesExtractor: false,
                featuresExtractor: .flatten
            )
        )
        let optimizerConfig = SACOptimizerConfig(
            actor: .adam(beta1: 0.85, beta2: 0.995, eps: 1e-7),
            critic: .adam(beta1: 0.88, beta2: 0.996, eps: 1e-7),
            entropy: .adam(beta1: 0.9, beta2: 0.999, eps: 1e-8)
        )
        let sac = try SAC(
            observationSpace: env.observationSpace,
            actionSpace: env.actionSpace,
            networksConfig: networksConfig,
            config: config,
            optimizerConfig: optimizerConfig,
            entCoef: .auto(init: 1.0),
            targetEntropy: -1.0,
            seed: 42
        )
        sac.setEnv(env)

        await sac.restore(
            timesteps: 75,
            totalTimesteps: 300,
            progressRemaining: 0.75,
            gradientSteps: 30
        )

        try await sac.save(to: dir)

        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("metadata.json").path))
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("policy.safetensors").path))
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("critic.safetensors").path))
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("critic_target.safetensors").path))
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("entropy.safetensors").path))

        let loaded = try SAC.load(from: dir, env: env)

        #expect(await loaded.numTimesteps == 75)
        #expect(await loaded.totalTimesteps == 300)
        #expect(abs(await loaded.progressRemaining - 0.75) < 1e-6)
        #expect(await loaded.gradientSteps == 30)
        #expect(await loaded.targetEntropy == -1.0)
        #expect(await loaded.entCoefConfig.isAuto)
        #expect(loaded.networksConfig == networksConfig)
        #expect(loaded.optimizerConfig == optimizerConfig)
        let loadedSacConfig = loaded.offPolicyConfig
        #expect(loadedSacConfig.bufferSize == 500)
        #expect(loadedSacConfig.batchSize == 32)

        let loadedNoEnv = try SAC.load(from: dir)
        #expect(loadedNoEnv.networksConfig == networksConfig)
        #expect(loadedNoEnv.optimizerConfig == optimizerConfig)
        var evalEnv = env
        let obs = try evalEnv.reset().obs
        let action = await loadedNoEnv(observation: obs, deterministic: true)
        #expect(action.size > 0)
    }

    @Test @MainActor
    func td3SaveLoad() async throws {
        let dir = tempDir()
        defer { cleanup(dir) }

        let env = try await Gymnazo.make("Pendulum")
        let offPolicyConfig = OffPolicyConfig(
            bufferSize: 500,
            learningStarts: 10,
            batchSize: 32,
            tau: 0.005,
            gamma: 0.99,
            trainFrequency: TrainFrequency(frequency: 1, unit: .step),
            gradientSteps: .fixed(1),
            targetUpdateInterval: 1,
            optimizeMemoryUsage: false,
            handleTimeoutTermination: true,
            useSDEAtWarmup: false,
            sdeSampleFreq: -1,
            sdeSupported: false
        )
        let policyConfig = TD3PolicyConfig(
            netArch: .shared([128, 64]),
            featuresExtractor: .flatten,
            activation: .relu,
            normalizeImages: true,
            nCritics: 2,
            shareFeaturesExtractor: false
        )
        let algorithmConfig = TD3AlgorithmConfig(
            policyDelay: 2,
            targetPolicyNoise: 0.2,
            targetNoiseClip: 0.5,
            actionNoise: .normal(std: 0.1)
        )
        let td3 = try TD3(
            observationSpace: env.observationSpace,
            actionSpace: env.actionSpace,
            learningRate: ConstantLearningRate(1e-3),
            policyConfig: policyConfig,
            algorithmConfig: algorithmConfig,
            config: offPolicyConfig,
            seed: 42
        )
        td3.setEnv(env)

        await td3.restore(
            timesteps: 90,
            totalTimesteps: 400,
            progressRemaining: 0.6,
            gradientSteps: 45
        )

        try await td3.save(to: dir)

        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("metadata.json").path))
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("policy.safetensors").path))
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("target.safetensors").path))
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("critic.safetensors").path))
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("critic_target.safetensors").path))

        let loaded = try TD3.load(from: dir, env: env)

        #expect(await loaded.numTimesteps == 90)
        #expect(await loaded.totalTimesteps == 400)
        #expect(abs(await loaded.progressRemaining - 0.6) < 1e-6)
        #expect(await loaded.gradientSteps == 45)
        #expect(loaded.policyConfig == policyConfig)
        #expect(loaded.algorithmConfig == algorithmConfig)
        #expect(loaded.offPolicyConfig.bufferSize == offPolicyConfig.bufferSize)
        #expect(loaded.offPolicyConfig.batchSize == offPolicyConfig.batchSize)
        #expect(abs(loaded.offPolicyConfig.tau - offPolicyConfig.tau) < 1e-9)
        #expect(abs(loaded.offPolicyConfig.gamma - offPolicyConfig.gamma) < 1e-9)
        #expect(loaded.offPolicyConfig.trainFrequency.frequency == offPolicyConfig.trainFrequency.frequency)
    }

    @Test @MainActor
    func ppoSaveLoad() async throws {
        let dir = tempDir()
        defer { cleanup(dir) }

        let env = try await Gymnazo.make("CartPole")
        let policyConfig = PPOPolicyConfig(
            netArch: .shared([64, 32]),
            featuresExtractor: .flatten,
            activation: .tanh,
            normalizeImages: true,
            shareFeaturesExtractor: true,
            orthoInit: false,
            logStdInit: 0.0,
            fullStd: true
        )
        let config = PPOConfig(
            nSteps: 32,
            batchSize: 8,
            nEpochs: 2,
            gamma: 0.99,
            gaeLambda: 0.95,
            clipRange: 0.2,
            clipRangeVf: nil,
            normalizeAdvantage: true,
            entCoef: 0.0,
            vfCoef: 0.5,
            maxGradNorm: 0.5,
            targetKL: 0.03,
            useSDE: false,
            sdeSampleFreq: -1
        )
        let ppo = try PPO(
            observationSpace: env.observationSpace,
            actionSpace: env.actionSpace,
            learningRate: ConstantLearningRate(3e-4),
            policyConfig: policyConfig,
            config: config,
            seed: 13
        )
        ppo.setEnv(env)

        await ppo.restore(
            timesteps: 60,
            totalTimesteps: 240,
            progressRemaining: 0.75,
            updates: 9
        )

        try await ppo.save(to: dir)

        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("metadata.json").path))
        #expect(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("policy.safetensors").path))

        let loaded = try PPO.load(from: dir, env: env)
        #expect(await loaded.numTimesteps == 60)
        #expect(await loaded.totalTimesteps == 240)
        #expect(abs(await loaded.progressRemaining - 0.75) < 1e-6)
        #expect(await loaded.nUpdates == 9)
        #expect(loaded.config == config)
        #expect(loaded.policyConfig == policyConfig)

        let loadedNoEnv = try PPO.load(from: dir)
        var evalEnv = try await Gymnazo.make("CartPole")
        let obs = try evalEnv.reset(seed: 99).obs
        let action = await loadedNoEnv(observation: obs, deterministic: true)
        #expect(action.size > 0)
    }

    @Test
    func checkpointMetadataRoundTrip() throws {
        let dir = tempDir()
        defer { cleanup(dir) }

        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let checkpoint = AlgorithmCheckpoint(
            algorithmKind: .dqn,
            numTimesteps: 1000,
            totalTimesteps: 5000,
            currentProgressRemaining: 0.8,
            learningRateSchedule: .constant(value: 1e-4),
            dqnConfig: DQNConfig(batchSize: 64),
            explorationRate: 0.5,
            numGradientSteps: 200
        )

        try checkpoint.write(to: dir)
        let loaded = try AlgorithmCheckpoint.read(from: dir)

        #expect(loaded.algorithmKind == .dqn)
        #expect(loaded.numTimesteps == 1000)
        #expect(loaded.totalTimesteps == 5000)
        #expect(abs(loaded.currentProgressRemaining - 0.8) < 1e-6)
        #expect(loaded.explorationRate == 0.5)
        #expect(loaded.numGradientSteps == 200)
        #expect(loaded.dqnConfig?.batchSize == 64)
    }

    @Test
    func learningRateScheduleRoundTrip() throws {
        let schedules: [(any LearningRateSchedule, Double)] = [
            (ConstantLearningRate(3e-4), 0.5),
            (LinearSchedule(initialValue: 1e-3, finalValue: 1e-5), 0.5),
            (ExponentialSchedule(initialValue: 1e-3, decayRate: 0.95), 0.5),
            (StepSchedule(initialValue: 1e-3, milestones: [0.3, 0.6], gamma: 0.1), 0.5),
            (CosineAnnealingSchedule(initialValue: 1e-3, minValue: 1e-5), 0.5),
        ]

        for (schedule, progress) in schedules {
            let originalValue = schedule.value(at: progress)
            guard let data = LearningRateScheduleData.from(schedule) else {
                Issue.record("Failed to serialize \(type(of: schedule))")
                continue
            }
            let restored = data.makeSchedule()
            let restoredValue = restored.value(at: progress)
            #expect(abs(originalValue - restoredValue) < 1e-10)
        }
    }

    @Test
    func replayBufferSaveLoad() throws {
        let dir = tempDir()
        defer { cleanup(dir) }

        let obsSpace = Box(low: MLXArray([-1.0, -1.0] as [Float]),
                           high: MLXArray([1.0, 1.0] as [Float]))
        let actSpace = Discrete(n: 4)
        let bufferConfig = ReplayBuffer.Configuration(
            bufferSize: 100,
            seed: 42
        )
        var buffer = ReplayBuffer(
            observationSpace: obsSpace,
            actionSpace: actSpace,
            config: bufferConfig
        )

        for i in 0..<10 {
            let obs = MLXArray([Float(i) * 0.1, Float(i) * -0.1])
            let action = MLXArray(Int32(i % 4))
            let reward = MLXArray(Float(i) * 0.5)
            let nextObs = MLXArray([Float(i + 1) * 0.1, Float(i + 1) * -0.1])
            buffer.add(
                obs: obs, action: action, reward: reward,
                nextObs: nextObs, terminated: false, truncated: false)
        }

        #expect(buffer.count == 10)

        try buffer.save(to: dir)

        var loadedBuffer = ReplayBuffer(
            observationSpace: obsSpace,
            actionSpace: actSpace,
            config: bufferConfig
        )
        try loadedBuffer.load(from: dir)

        #expect(loadedBuffer.count == 10)

        for i in 0..<10 {
            let origObs = buffer.observations[i]
            let loadedObs = loadedBuffer.observations[i]
            let diff = MLX.abs(origObs - loadedObs)
            eval(diff)
            #expect(MLX.max(diff).item(Float.self) < 1e-6)
        }
    }
}
