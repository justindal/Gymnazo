//
//  TabularConfig.swift
//  Gymnazo
//

public struct TabularConfig: Sendable, Codable {
    public let learningRate: Double
    public let gamma: Double
    public let epsilon: Double
    public let epsilonDecay: Double
    public let minEpsilon: Double

    public init(
        learningRate: Double = 0.1,
        gamma: Double = 0.99,
        epsilon: Double = 1.0,
        epsilonDecay: Double = 0.999,
        minEpsilon: Double = 0.05
    ) {
        self.learningRate = learningRate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.minEpsilon = minEpsilon
    }
}
