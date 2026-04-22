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

    public static func load(
        from directory: URL,
        env: (any Env)? = nil
    ) throws -> PPO {
        let checkpoint = try AlgorithmCheckpoint.read(from: directory)

        guard checkpoint.algorithmKind == .ppo else {
            throw PersistenceError.invalidCheckpoint(
                "Expected PPO checkpoint, got \(checkpoint.algorithmKind)"
            )
        }

        let config = checkpoint.ppoConfig ?? PPOConfig()
        let policyConfig = checkpoint.ppoPolicyConfig ?? PPOPolicyConfig()
        let learningRate =
            checkpoint.learningRateSchedule?.makeSchedule()
            ?? ConstantLearningRate(3e-4)

        let observationSpace: any Space
        let actionSpace: any Space
        if let env {
            observationSpace = env.observationSpace
            actionSpace = env.actionSpace
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
        let optimizer = optimizerConfig.make(
            learningRate: Float(learningRate.value(at: checkpoint.currentProgressRemaining))
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
            updates: checkpoint.numGradientSteps ?? 0
        )

        if let env {
            agent.setEnv(env)
        }

        return agent
    }
}
