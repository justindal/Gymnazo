//
//  DQNConfigs.swift
//  Gymnazo
//

import MLX
import MLXNN
import MLXOptimizers

/// Configuration for the DQN Q-Network.
public struct DQNPolicyConfig: Sendable, Codable, Equatable {
    public var netArch: [Int]
    public var featuresExtractor: FeaturesExtractorConfig
    public var activation: ActivationConfig
    public var normalizeImages: Bool

    public init(
        netArch: [Int] = [64, 64],
        featuresExtractor: FeaturesExtractorConfig = .auto,
        activation: ActivationConfig = .relu,
        normalizeImages: Bool = true
    ) {
        self.netArch = netArch
        self.featuresExtractor = featuresExtractor
        self.activation = activation
        self.normalizeImages = normalizeImages
    }
}

/// Configuration for DQN hyperparameters.
public struct DQNConfig: Sendable, Codable {
    public let bufferSize: Int
    public let learningStarts: Int
    public let batchSize: Int
    public let tau: Double
    public let gamma: Double
    public let trainFrequency: TrainFrequency
    public let gradientSteps: GradientSteps
    public let targetUpdateInterval: Int
    public let explorationFraction: Double
    public let explorationInitialEps: Double
    public let explorationFinalEps: Double
    public let maxGradNorm: Double?
    public let optimizeMemoryUsage: Bool
    public let handleTimeoutTermination: Bool

    public init(
        bufferSize: Int = 1_000_000,
        learningStarts: Int = 100,
        batchSize: Int = 32,
        tau: Double = 1.0,
        gamma: Double = 0.99,
        trainFrequency: TrainFrequency = TrainFrequency(frequency: 4, unit: .step),
        gradientSteps: GradientSteps = .fixed(1),
        targetUpdateInterval: Int = 10_000,
        explorationFraction: Double = 0.1,
        explorationInitialEps: Double = 1.0,
        explorationFinalEps: Double = 0.05,
        maxGradNorm: Double? = 10.0,
        optimizeMemoryUsage: Bool = false,
        handleTimeoutTermination: Bool = true
    ) {
        let safeTrainFrequency = TrainFrequency(
            frequency: max(1, trainFrequency.frequency),
            unit: trainFrequency.unit
        )
        let safeGradientSteps: GradientSteps
        switch gradientSteps {
        case .fixed(let steps):
            safeGradientSteps = .fixed(max(1, steps))
        case .asCollectedSteps:
            safeGradientSteps = .asCollectedSteps
        }
        let safeInitialEps = min(max(explorationInitialEps, 0.0), 1.0)
        let safeFinalEps = min(
            max(explorationFinalEps, 0.0),
            safeInitialEps
        )
        let safeOptimizeMemoryUsage =
            optimizeMemoryUsage && handleTimeoutTermination
            ? false : optimizeMemoryUsage

        self.bufferSize = max(1, bufferSize)
        self.learningStarts = max(0, learningStarts)
        self.batchSize = max(1, batchSize)
        self.tau = min(max(tau, 0.0), 1.0)
        self.gamma = min(max(gamma, 0.0), 1.0)
        self.trainFrequency = safeTrainFrequency
        self.gradientSteps = safeGradientSteps
        self.targetUpdateInterval = max(1, targetUpdateInterval)
        self.explorationFraction = min(max(explorationFraction, 1e-9), 1.0)
        self.explorationInitialEps = safeInitialEps
        self.explorationFinalEps = safeFinalEps
        self.maxGradNorm = maxGradNorm.map { max(0.0, $0) }
        self.optimizeMemoryUsage = safeOptimizeMemoryUsage
        self.handleTimeoutTermination = handleTimeoutTermination
    }
}

/// Optimizer configuration for DQN.
public struct DQNOptimizerConfig: Sendable, Codable, Equatable {
    public var optimizer: OptimizerConfig

    public init(optimizer: OptimizerConfig = .adam()) {
        self.optimizer = optimizer
    }
}
