//
//  TD3.swift
//  Gymnazo
//
//  Created by Justin Daludado on 2026-02-24.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

extension TD3 {
    public func save(to directory: URL) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try actor.saveWeights(to: directory.appendingPathComponent(CheckpointFiles.policy))
        try targetActor.saveWeights(to: directory.appendingPathComponent(CheckpointFiles.target))
        try critic.saveWeights(to: directory.appendingPathComponent(CheckpointFiles.critic))
        try targetCritic.saveWeights(
            to: directory.appendingPathComponent(CheckpointFiles.criticTarget))

        if let buffer {
            let bufferDir = directory.appendingPathComponent(CheckpointFiles.bufferDirectory)
            try buffer.save(to: bufferDir)
        }

        let checkpoint = AlgorithmCheckpoint(
            algorithmKind: .td3,
            numTimesteps: numTimesteps,
            totalTimesteps: totalTimesteps,
            currentProgressRemaining: progressRemaining,
            learningRateSchedule: LearningRateScheduleData.from(learningRate),
            offPolicyConfig: offPolicyConfig,
            td3PolicyConfig: policyConfig,
            td3AlgorithmConfig: algorithmConfig,
            numGradientSteps: gradientSteps,
            seed: randomSeed
        )
        try checkpoint.write(to: directory)
    }

    public static func load(
        from directory: URL,
        env: sending (any Env)? = nil,
        includeBuffer: Bool = true
    ) async throws -> TD3 {
        let checkpoint = try AlgorithmCheckpoint.read(from: directory)

        guard checkpoint.algorithmKind == .td3 else {
            throw PersistenceError.invalidCheckpoint(
                "Expected TD3 checkpoint, got \(checkpoint.algorithmKind)"
            )
        }

        guard let offPolicyConfig = checkpoint.offPolicyConfig else {
            throw PersistenceError.invalidCheckpoint("Missing off-policy config")
        }

        guard let environment = env else {
            throw PersistenceError.invalidCheckpoint(
                "TD3.load requires an environment to reconstruct network architecture"
            )
        }

        let schedule =
            checkpoint.learningRateSchedule?.makeSchedule()
            ?? ConstantLearningRate(1e-3)
        let policyConfig = checkpoint.td3PolicyConfig ?? TD3PolicyConfig()
        let algorithmConfig = checkpoint.td3AlgorithmConfig ?? TD3AlgorithmConfig()
        let observationSpace =
            try SpaceDescriptor
            .from(space: environment.observationSpace)
            .makeSpace()
        let actionSpace =
            try SpaceDescriptor
            .from(space: environment.actionSpace)
            .makeSpace()

        let policy = try TD3Policy(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            learningRateSchedule: schedule,
            config: policyConfig
        )

        try policy.actor.loadWeights(
            from: directory.appendingPathComponent(CheckpointFiles.policy)
        )
        try policy.actorTarget.loadWeights(
            from: directory.appendingPathComponent(CheckpointFiles.target)
        )
        try policy.critic.loadWeights(
            from: directory.appendingPathComponent(CheckpointFiles.critic)
        )
        try policy.criticTarget.loadWeights(
            from: directory.appendingPathComponent(CheckpointFiles.criticTarget)
        )
        eval(
            policy.actor.parameters(),
            policy.actorTarget.parameters(),
            policy.critic.parameters(),
            policy.criticTarget.parameters()
        )

        var replayBuffer: ReplayBuffer?
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
                var buffer = try ReplayBuffer(
                    observationSpace:
                        try SpaceDescriptor
                        .from(space: policy.observationSpace)
                        .makeSpace(),
                    actionSpace:
                        try SpaceDescriptor
                        .from(space: policy.actionSpace)
                        .makeSpace(),
                    config: bufferConfig,
                    numEnvs: 1
                )
                try buffer.load(from: bufferDir)
                replayBuffer = buffer
            }
        }

        let agent = TD3(
            policy: policy,
            offPolicyConfig: offPolicyConfig,
            policyConfig: policyConfig,
            algorithmConfig: algorithmConfig,
            learningRate: schedule,
            seed: checkpoint.seed,
            timesteps: checkpoint.numTimesteps,
            totalTimesteps: checkpoint.totalTimesteps,
            progressRemaining: checkpoint.currentProgressRemaining,
            gradientSteps: checkpoint.numGradientSteps ?? 0,
            buffer: replayBuffer,
            env: nil
        )

        if let env {
            try await agent.setEnv(env)
        }

        return agent
    }
}
