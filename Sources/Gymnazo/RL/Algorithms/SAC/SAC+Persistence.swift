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
            seed: randomSeed
        )
        try checkpoint.write(to: directory)
    }

    public static func load(
        from directory: URL,
        env: (any Env)? = nil,
        includeBuffer: Bool = true
    ) throws -> SAC {
        let checkpoint = try AlgorithmCheckpoint.read(from: directory)

        guard checkpoint.algorithmKind == .sac else {
            throw PersistenceError.invalidCheckpoint(
                "Expected SAC checkpoint, got \(checkpoint.algorithmKind)")
        }

        guard let offPolicyConfig = checkpoint.offPolicyConfig else {
            throw PersistenceError.invalidCheckpoint("Missing off-policy config")
        }

        guard let environment = env else {
            throw PersistenceError.invalidCheckpoint(
                "SAC.load requires an environment to reconstruct network architecture")
        }

        let schedule = checkpoint.learningRateSchedule?.makeSchedule()
            ?? ConstantLearningRate(3e-4)
        let entCoef = checkpoint.entCoefConfig ?? .auto()

        let networks = SACNetworks(
            observationSpace: environment.observationSpace,
            actionSpace: environment.actionSpace
        )

        try networks.actor.loadWeights(
            from: directory.appendingPathComponent(CheckpointFiles.policy))
        try networks.critic.loadWeights(
            from: directory.appendingPathComponent(CheckpointFiles.critic))
        try networks.criticTarget.loadWeights(
            from: directory.appendingPathComponent(CheckpointFiles.criticTarget))

        let lr = Float(schedule.value(at: checkpoint.currentProgressRemaining))
        let optimizerConfig = SACOptimizerConfig()
        let actorOpt = optimizerConfig.actor.make(learningRate: lr)
        let criticOpt = optimizerConfig.critic.make(learningRate: lr)
        let entOpt: Adam? = entCoef.isAuto ? optimizerConfig.entropy?.make(learningRate: lr) : nil

        let entModule = LogEntropyCoefModule(initialValue: entCoef.initialValue)
        try entModule.loadWeights(
            from: directory.appendingPathComponent(CheckpointFiles.entropy))

        eval(
            networks.actor.parameters(),
            networks.critic.parameters(),
            networks.criticTarget.parameters(),
            entModule.parameters()
        )

        var buf: ReplayBuffer?
        if includeBuffer {
            let bufferDir = directory.appendingPathComponent(CheckpointFiles.bufferDirectory)
            if FileManager.default.fileExists(atPath: bufferDir.path) {
                let bufferConfig = ReplayBuffer.Configuration(
                    bufferSize: offPolicyConfig.bufferSize,
                    optimizeMemoryUsage: offPolicyConfig.optimizeMemoryUsage,
                    handleTimeoutTermination: offPolicyConfig.handleTimeoutTermination,
                    seed: checkpoint.seed
                )
                var buffer = ReplayBuffer(
                    observationSpace: networks.actor.observationSpace,
                    actionSpace: networks.actor.actionSpace,
                    config: bufferConfig,
                    numEnvs: 1
                )
                try buffer.load(from: bufferDir)
                buf = buffer
            }
        }

        let actionDim = getActionDim(environment.actionSpace)

        let agent = SAC(
            policy: networks.actor,
            critic: networks.critic,
            criticTarget: networks.criticTarget,
            actorOptimizer: actorOpt,
            criticOptimizer: criticOpt,
            entropyOptimizer: entOpt,
            logEntCoefModule: entModule,
            offPolicyConfig: offPolicyConfig,
            entCoefConfig: entCoef,
            targetEntropy: checkpoint.targetEntropy ?? Float(-actionDim),
            learningRate: schedule,
            seed: checkpoint.seed,
            timesteps: checkpoint.numTimesteps,
            totalTimesteps: checkpoint.totalTimesteps,
            progressRemaining: checkpoint.currentProgressRemaining,
            gradientSteps: checkpoint.numGradientSteps ?? 0,
            buffer: buf
        )

        if let env {
            agent.setEnv(env)
        }

        return agent
    }
}
