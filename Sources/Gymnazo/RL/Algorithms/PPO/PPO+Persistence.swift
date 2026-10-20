import Foundation
import MLX
import MLXNN
import MLXOptimizers

extension PPO {
    public func save(to directory: URL) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try policy.saveWeights(to: directory.appendingPathComponent(CheckpointFiles.policy))

        let observationSpace = try SpaceDescriptor.from(space: policy.observationSpace)
        let actionSpace = try SpaceDescriptor.from(space: policy.actionSpace)

        let checkpoint = AlgorithmCheckpoint(
            algorithmKind: .ppo,
            numTimesteps: numTimesteps,
            totalTimesteps: totalTimesteps,
            currentProgressRemaining: progressRemaining,
            learningRateSchedule: LearningRateScheduleData.from(learningRate),
            ppoConfig: config,
            ppoPolicyConfig: policyConfig,
            ppoObservationSpace: observationSpace,
            ppoActionSpace: actionSpace,
            numGradientSteps: updates,
            seed: randomSeed
        )
        try checkpoint.write(to: directory)
    }

    /// Loads learned weights and counters, optionally replacing learning settings.
    /// Optimizer state is initialized anew. Policy architecture is retained from the checkpoint.
    public static func load(
        from directory: URL,
        env: sending (any Env)? = nil,
        config overrideConfig: PPOConfig? = nil,
        learningRate overrideLearningRate: (any LearningRateSchedule)? = nil
    ) async throws -> PPO {
        let checkpoint = try AlgorithmCheckpoint.read(from: directory)

        guard checkpoint.algorithmKind == .ppo else {
            throw PersistenceError.invalidCheckpoint(
                "Expected PPO checkpoint, got \(checkpoint.algorithmKind)"
            )
        }

        let config = overrideConfig ?? checkpoint.ppoConfig ?? PPOConfig()
        let policyConfig = checkpoint.ppoPolicyConfig ?? PPOPolicyConfig()
        let learningRate =
            overrideLearningRate ?? checkpoint.learningRateSchedule?.makeSchedule()
            ?? ConstantLearningRate(3e-4)

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
            guard let ppoObservationSpace = checkpoint.ppoObservationSpace,
                let ppoActionSpace = checkpoint.ppoActionSpace
            else {
                throw PersistenceError.invalidCheckpoint(
                    "PPO.load without env requires checkpointed observation/action spaces"
                )
            }
            observationSpace = try ppoObservationSpace.makeSpace()
            actionSpace = try ppoActionSpace.makeSpace()
        }

        let policy = try PPOPolicy(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            config: policyConfig,
            useSDE: config.useSDE
        )
        try policy.loadWeights(
            from: directory.appendingPathComponent(CheckpointFiles.policy)
        )
        eval(policy.parameters())

        let optimizerConfig: OptimizerConfig = .adam()
        let optimizer = optimizerConfig.makeScheduled(
            learningRate: Float(learningRate.value(at: checkpoint.currentProgressRemaining)),
            schedule: learningRate
        )

        let agent = PPO(
            policy: policy,
            optimizer: optimizer,
            config: config,
            policyConfig: policyConfig,
            learningRate: learningRate,
            seed: checkpoint.seed,
            timesteps: checkpoint.numTimesteps,
            totalTimesteps: checkpoint.totalTimesteps,
            progressRemaining: checkpoint.currentProgressRemaining,
            updates: checkpoint.numGradientSteps ?? 0,
            env: nil
        )

        if let env {
            try await agent.setEnv(env)
        }

        return agent
    }
}
