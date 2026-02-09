//
//  OffPolicyAlgorithm.swift
//  Gymnazo
//

public enum TrainFrequencyUnit: String, Sendable, Codable {
    case step
    case episode
}

public struct TrainFrequency: Sendable, Codable {
    public let frequency: Int
    public let unit: TrainFrequencyUnit

    public init(frequency: Int, unit: TrainFrequencyUnit = .step) {
        self.frequency = frequency
        self.unit = unit
    }
}

public enum GradientSteps: Sendable, Codable {
    case fixed(Int)
    case asCollectedSteps
}

public struct OffPolicyConfig: Sendable, Codable {
    public let bufferSize: Int
    public let learningStarts: Int
    public let batchSize: Int
    public let tau: Double
    public let gamma: Double
    public let trainFrequency: TrainFrequency
    public let gradientSteps: GradientSteps
    public let targetUpdateInterval: Int
    public let optimizeMemoryUsage: Bool
    public let handleTimeoutTermination: Bool
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
        targetUpdateInterval: Int = 1,
        optimizeMemoryUsage: Bool = false,
        handleTimeoutTermination: Bool = true,
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
        self.targetUpdateInterval = targetUpdateInterval
        self.optimizeMemoryUsage = optimizeMemoryUsage
        self.handleTimeoutTermination = handleTimeoutTermination
        self.useSDEAtWarmup = useSDEAtWarmup
        self.sdeSampleFreq = sdeSampleFreq
        self.sdeSupported = sdeSupported
    }
}
