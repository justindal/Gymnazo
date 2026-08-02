//
//  DQN+Persistence.swift
//  Gymnazo
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

extension DQN {
    public func save(to directory: URL) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try qNet.saveWeights(to: directory.appendingPathComponent(CheckpointFiles.policy))
        try qNetTarget.saveWeights(to: directory.appendingPathComponent(CheckpointFiles.target))

        if let buffer {
            let bufferDir = directory.appendingPathComponent(CheckpointFiles.bufferDirectory)
            try buffer.save(to: bufferDir)
        }

        let checkpoint = AlgorithmCheckpoint(
            algorithmKind: .dqn,
            numTimesteps: numTimesteps,
            totalTimesteps: totalTimesteps,
            currentProgressRemaining: progressRemaining,
            learningRateSchedule: LearningRateScheduleData.from(learningRate),
            dqnConfig: config,
            dqnPolicyConfig: policyConfig,
            dqnOptimizerConfig: optimizerConfig,
            explorationRate: explorationRate,
            numGradientSteps: gradientSteps,
            seed: randomSeed
        )
        try checkpoint.write(to: directory)
    }

    public static func load(
        from directory: URL,
        env: sending (any Env)? = nil,
        includeBuffer: Bool = true
    ) async throws -> DQN {
        let checkpoint = try AlgorithmCheckpoint.read(from: directory)

        guard checkpoint.algorithmKind == .dqn else {
            throw PersistenceError.invalidCheckpoint(
                "Expected DQN checkpoint, got \(checkpoint.algorithmKind)")
        }

        guard let dqnConfig = checkpoint.dqnConfig else {
            throw PersistenceError.invalidCheckpoint("Missing DQN config")
        }
        let policyConfig = checkpoint.dqnPolicyConfig ?? DQNPolicyConfig()
        let optimizerConfig = checkpoint.dqnOptimizerConfig ?? DQNOptimizerConfig()

        guard let environment = env else {
            throw PersistenceError.invalidCheckpoint(
                "DQN.load requires an environment to reconstruct network architecture")
        }

        let observationSpace =
            try SpaceDescriptor
            .from(space: environment.observationSpace)
            .makeSpace()
        guard
            let discrete = try SpaceDescriptor
                .from(space: environment.actionSpace)
                .makeSpace() as? Discrete
        else {
            throw PersistenceError.invalidCheckpoint("DQN requires a Discrete action space")
        }

        let schedule =
            checkpoint.learningRateSchedule?.makeSchedule()
            ?? ConstantLearningRate(1e-4)

        let networks = try DQNNetworks(
            observationSpace: observationSpace,
            nActions: discrete.n,
            config: policyConfig
        )

        try networks.qNet.loadWeights(
            from: directory.appendingPathComponent(CheckpointFiles.policy))
        try networks.qNetTarget.loadWeights(
            from: directory.appendingPathComponent(CheckpointFiles.target))
        eval(networks.qNet.parameters(), networks.qNetTarget.parameters())

        let optimizer = optimizerConfig.optimizer.make(
            learningRate: Float(schedule.value(at: checkpoint.currentProgressRemaining))
        )

        var buf: ReplayBuffer?
        if includeBuffer {
            let bufferDir = directory.appendingPathComponent(CheckpointFiles.bufferDirectory)
            if FileManager.default.fileExists(atPath: bufferDir.path) {
                let bufferConfig = ReplayBuffer.Configuration(
                    bufferSize: dqnConfig.bufferSize,
                    optimizeMemoryUsage: dqnConfig.optimizeMemoryUsage,
                    handleTimeoutTermination: dqnConfig.handleTimeoutTermination,
                    frameStack: dqnConfig.replayFrameStack,
                    seed: checkpoint.seed
                )
                var buffer = try ReplayBuffer(
                    observationSpace:
                        try SpaceDescriptor
                        .from(space: networks.qNet.observationSpace)
                        .makeSpace(),
                    actionSpace:
                        try SpaceDescriptor
                        .from(space: networks.qNet.actionSpace)
                        .makeSpace(),
                    config: bufferConfig,
                    numEnvs: 1
                )
                try buffer.load(from: bufferDir)
                buf = buffer
            }
        }

        let agent = DQN(
            networks: networks,
            optimizer: optimizer,
            config: dqnConfig,
            policyConfig: policyConfig,
            optimizerConfig: optimizerConfig,
            learningRate: schedule,
            seed: checkpoint.seed,
            timesteps: checkpoint.numTimesteps,
            totalTimesteps: checkpoint.totalTimesteps,
            progressRemaining: checkpoint.currentProgressRemaining,
            explorationRate: checkpoint.explorationRate ?? dqnConfig.explorationInitialEps,
            gradientSteps: checkpoint.numGradientSteps ?? 0,
            buffer: buf,
            env: nil
        )

        try await agent.setEnv(environment)

        return agent
    }
}
