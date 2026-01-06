//
//  OffPolicyAlgorithm.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Protocol for Off-Policy Algorithms like SAC or TD3.
///
/// - Parameters:
///     - config: Hyperparameters that define the off-policy learning loop.
///     - replayBuffer: Replay buffer for storing experiences.
///     - actionNoise: Optional action noise for exploration.
public protocol OffPolicyAlgorithm: Algorithm {
    var config: OffPolicyConfig { get }
    var replayBuffer: (any ReplayBuffer)? { get set }
    var actionNoise: (any ActionNoise)? { get set }

    mutating func train(gradientSteps: Int, batchSize: Int)
}

/// Units for training frequency.
public enum TrainFrequencyUnit: String, Sendable {
    case step
    case episode
}

/// Training frequency for off-policy algorithms.
public struct TrainFrequency: Sendable {
    public let frequency: Int
    public let unit: TrainFrequencyUnit

    public init(frequency: Int, unit: TrainFrequencyUnit = .step) {
        self.frequency = frequency
        self.unit = unit
    }
}

/// How many gradient steps to run after collecting data.
public enum GradientSteps: Sendable {
    case fixed(Int)
    case asCollectedSteps
}

/// Hyperparameters for off-policy algorithms.
public struct OffPolicyConfig: Sendable {
    public let bufferSize: Int
    public let learningStarts: Int
    public let batchSize: Int
    public let tau: Double
    public let gamma: Double
    public let trainFrequency: TrainFrequency
    public let gradientSteps: GradientSteps
    public let optimizeMemoryUsage: Bool
    public let nSteps: Int
    public let useSDEAtWarmup: Bool
    public let sdeSampleFreq: Int
    public let sdeSupported: Bool

    public init(
        bufferSize: Int = 1_000_000,
        learningStarts: Int = 100,
        batchSize: Int = 256,
        tau: Double = 0.005,
        gamma: Double = 0.99,
        trainFrequency: TrainFrequency = TrainFrequency(frequency: 1, unit: .step),
        gradientSteps: GradientSteps = .fixed(1),
        optimizeMemoryUsage: Bool = false,
        nSteps: Int = 1,
        useSDEAtWarmup: Bool = false,
        sdeSampleFreq: Int = -1,
        sdeSupported: Bool = true
    ) {
        self.bufferSize = bufferSize
        self.learningStarts = learningStarts
        self.batchSize = batchSize
        self.tau = tau
        self.gamma = gamma
        self.trainFrequency = trainFrequency
        self.gradientSteps = gradientSteps
        self.optimizeMemoryUsage = optimizeMemoryUsage
        self.nSteps = nSteps
        self.useSDEAtWarmup = useSDEAtWarmup
        self.sdeSampleFreq = sdeSampleFreq
        self.sdeSupported = sdeSupported
    }
}

/// Action noise for exploration in continuous action spaces.
public protocol ActionNoise {
    mutating func reset()
    mutating func sample() -> MLXArray
}
