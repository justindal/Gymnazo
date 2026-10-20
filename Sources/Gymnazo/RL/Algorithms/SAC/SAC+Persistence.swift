//
//  SAC+Persistence.swift
//  Gymnazo
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

extension SAC {
    public func save(to directory: URL) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try policy.saveWeights(to: directory.appendingPathComponent(CheckpointFiles.policy))
        try critic.saveWeights(to: directory.appendingPathComponent(CheckpointFiles.critic))
        try criticTarget.saveWeights(
            to: directory.appendingPathComponent(CheckpointFiles.criticTarget))
        try logEntCoefModule.saveWeights(
            to: directory.appendingPathComponent(CheckpointFiles.entropy))

        if let buffer {
            let bufferDir = directory.appendingPathComponent(CheckpointFiles.bufferDirectory)
            try buffer.save(to: bufferDir)
        }

        let observationSpace = try SpaceDescriptor.from(space: policy.observationSpace)
        let actionSpace = try SpaceDescriptor.from(space: policy.actionSpace)

        let checkpoint = AlgorithmCheckpoint(
            algorithmKind: .sac,
            numTimesteps: numTimesteps,
            totalTimesteps: totalTimesteps,
            currentProgressRemaining: progressRemaining,
            learningRateSchedule: LearningRateScheduleData.from(learningRate),
            offPolicyConfig: offPolicyConfig,
            explorationRate: nil,
            numGradientSteps: gradientSteps,
            targetEntropy: targetEntropy,
            entCoefConfig: entCoefConfig,
            sacNetworksConfig: networksConfig,
            sacOptimizerConfig: optimizerConfig,
            sacObservationSpace: observationSpace,
            sacActionSpace: actionSpace,
            seed: randomSeed
        )
        try checkpoint.write(to: directory)
    }

    /// Loads learned weights and counters, optionally replacing learning settings.
    /// Optimizer state is initialized anew. Policy architecture is retained from the checkpoint.
    /// Restoring replay data requires unchanged capacity, storage mode and frame stacking.
    public static func load(
        from directory: URL,
        env: sending (any Env)? = nil,
        includeBuffer: Bool = true,
        config overrideConfig: OffPolicyConfig? = nil,
        learningRate overrideLearningRate: (any LearningRateSchedule)? = nil
    ) async throws -> SAC {
        let checkpoint = try AlgorithmCheckpoint.read(from: directory)

        guard checkpoint.algorithmKind == .sac else {
            throw PersistenceError.invalidCheckpoint(
                "Expected SAC checkpoint, got \(checkpoint.algorithmKind)")
        }

        if includeBuffer, let proposed = overrideConfig, let saved = checkpoint.offPolicyConfig {
            guard proposed.bufferSize == saved.bufferSize,
                proposed.optimizeMemoryUsage == saved.optimizeMemoryUsage,
                proposed.handleTimeoutTermination == saved.handleTimeoutTermination,
                proposed.replayFrameStack == saved.replayFrameStack else {
                throw PersistenceError.invalidCheckpoint("Replay layout changes require loading without replay data.")
            }
        }
        guard let offPolicyConfig = overrideConfig ?? checkpoint.offPolicyConfig else {
            throw PersistenceError.invalidCheckpoint("Missing off-policy config")
        }

        let schedule =
            overrideLearningRate ?? checkpoint.learningRateSchedule?.makeSchedule()
            ?? ConstantLearningRate(3e-4)
        let entCoef = checkpoint.entCoefConfig ?? .auto()
        let networksConfig = checkpoint.sacNetworksConfig ?? SACNetworksConfig()
        let optimizerConfig = checkpoint.sacOptimizerConfig ?? SACOptimizerConfig()

        let observationSpace: any Space
        let actionSpace: any Space
        if let env {
            observationSpace =
                try SpaceDescriptor
                .from(space: env.observationSpace)
                .makeSpace()
            actionSpace =
                try SpaceDescriptor
                .from(space: env.actionSpace)
                .makeSpace()
        } else {
            guard let observationDescriptor = checkpoint.sacObservationSpace,
                let actionDescriptor = checkpoint.sacActionSpace
            else {
                throw PersistenceError.invalidCheckpoint(
                    "SAC.load without env requires checkpointed observation/action spaces"
                )
            }
            observationSpace = try observationDescriptor.makeSpace()
            actionSpace = try actionDescriptor.makeSpace()
        }

        let agent = try SAC(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            env: env,
            learningRate: schedule,
            networksConfig: networksConfig,
            config: offPolicyConfig,
            optimizerConfig: optimizerConfig,
            entCoef: entCoef,
            targetEntropy: checkpoint.targetEntropy,
            seed: checkpoint.seed
        )
        try await agent.restore(
            from: directory, checkpoint: checkpoint, includeBuffer: includeBuffer)

        return agent
    }

    private func restore(
        from directory: URL,
        checkpoint: AlgorithmCheckpoint,
        includeBuffer: Bool
    ) throws {
        try policy.loadWeights(from: directory.appendingPathComponent(CheckpointFiles.policy))
        try critic.loadWeights(from: directory.appendingPathComponent(CheckpointFiles.critic))
        try criticTarget.loadWeights(
            from: directory.appendingPathComponent(CheckpointFiles.criticTarget))
        try logEntCoefModule.loadWeights(
            from: directory.appendingPathComponent(CheckpointFiles.entropy))
        eval(
            policy.parameters(), critic.parameters(), criticTarget.parameters(),
            logEntCoefModule.parameters())

        if includeBuffer {
            let bufferDir = directory.appendingPathComponent(CheckpointFiles.bufferDirectory)
            if FileManager.default.fileExists(atPath: bufferDir.path) {
                let bufferConfig = ReplayBuffer.Configuration(
                    bufferSize: offPolicyConfig.bufferSize,
                    optimizeMemoryUsage: offPolicyConfig.optimizeMemoryUsage,
                    handleTimeoutTermination: offPolicyConfig.handleTimeoutTermination,
                    frameStack: offPolicyConfig.replayFrameStack,
                    seed: checkpoint.seed
                )
                var restoredBuffer = try ReplayBuffer(
                    observationSpace: policy.observationSpace,
                    actionSpace: policy.actionSpace,
                    config: bufferConfig,
                    numEnvs: 1
                )
                try restoredBuffer.load(from: bufferDir)
                buffer = restoredBuffer
            }
        }

        timesteps = checkpoint.numTimesteps
        totalTimesteps = checkpoint.totalTimesteps
        progressRemaining = checkpoint.currentProgressRemaining
        gradientSteps = checkpoint.numGradientSteps ?? 0
    }
}
