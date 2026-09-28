import Gymnazo
import MLX

extension GymnazoProfiler {
    static func profileVectorTraining(vectorEnvironments: Int) async throws -> [ProfileMeasurement]
    {
        let model = try DQN(
            observationSpace: Box(low: -100, high: 100, shape: [4]), actionSpace: Discrete(n: 2),
            policyConfig: DQNPolicyConfig(netArch: [64, 64], featuresExtractor: .flatten),
            config: DQNConfig(
                bufferSize: 2_048, learningStarts: 32, batchSize: 32,
                trainFrequency: TrainFrequency(frequency: 1, unit: .step),
                gradientSteps: .fixed(vectorEnvironments), targetUpdateInterval: 64,
                explorationInitialEps: 0, explorationFinalEps: 0
            ), seed: 19
        )
        let environments = try (0..<vectorEnvironments).map { _ in try CartPoleValueEnvironment() }
        let warm = try await measured(name: "vectorTrainingWarmUp", operations: 256) {
            try await model.learn(environments: environments, totalTimesteps: 256)
        }
        let steady = try await measured(name: "vectorTrainingSteadyState", operations: 2_048) {
            try await model.learn(
                environments: environments, totalTimesteps: 2_304, resetProgress: false)
        }
        return [warm.measurement, steady.measurement]
    }

    static func profilePPO() async throws -> [ProfileMeasurement] {
        let model = try PPO(
            env: CartPole(),
            policyConfig: PPOPolicyConfig(netArch: .shared([64, 64]), featuresExtractor: .flatten),
            config: PPOConfig(nSteps: 128, batchSize: 32, nEpochs: 4), seed: 19
        )
        let warm = try await measured(name: "ppoTrainingWarmUp", operations: 512) {
            try await model.learn(totalTimesteps: 512, callbacks: nil)
        }
        let steady = try await measured(name: "ppoTrainingSteadyState", operations: 2_048) {
            try await model.learn(totalTimesteps: 2_560, callbacks: nil, resetProgress: false)
        }
        return [warm.measurement, steady.measurement]
    }

    static func profileCNN() async throws -> [ProfileMeasurement] {
        let model = try DQN(
            env: ProfileImageEnvironment(),
            policyConfig: DQNPolicyConfig(
                netArch: [32], featuresExtractor: .natureCNN(featuresDim: 32)),
            config: DQNConfig(bufferSize: 64, learningStarts: 4, batchSize: 4), seed: 19
        )
        let warm = try await measured(name: "cnnTrainingWarmUp", operations: 64) {
            try await model.learn(totalTimesteps: 64, callbacks: nil)
        }
        let steady = try await measured(name: "cnnTrainingSteadyState", operations: 256) {
            try await model.learn(totalTimesteps: 320, callbacks: nil, resetProgress: false)
        }
        return [warm.measurement, steady.measurement]
    }
}

private struct ProfileImageEnvironment: Env {
    let observationSpace: any Space = Box(low: 0, high: 1, shape: [36, 36, 1])
    let actionSpace: any Space = Discrete(n: 2)
    var spec: EnvSpec?
    var renderMode: RenderMode?

    mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        Reset(obs: MLX.ones([36, 36, 1]) * 0.5)
    }

    mutating func step(_ action: MLXArray) throws -> Step {
        Step(obs: MLX.ones([36, 36, 1]) * 0.25, reward: 1, terminated: true, truncated: false)
    }
}
