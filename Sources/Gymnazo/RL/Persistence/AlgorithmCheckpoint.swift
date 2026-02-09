//
//  AlgorithmCheckpoint.swift
//  Gymnazo
//

import Foundation

/// Identifies the type of algorithm stored in a checkpoint.
public enum AlgorithmKind: String, Codable, Sendable {
    case sac
    case dqn
    case qLearning
    case sarsa
}

/// Metadata stored alongside model weights in a checkpoint directory.
public struct AlgorithmCheckpoint: Codable, Sendable {
    public let version: String
    public let algorithmKind: AlgorithmKind
    public let numTimesteps: Int
    public let totalTimesteps: Int
    public let currentProgressRemaining: Double

    public let learningRateSchedule: LearningRateScheduleData?

    public let offPolicyConfig: OffPolicyConfig?
    public let dqnConfig: DQNConfig?
    public let dqnPolicyConfig: DQNPolicyConfig?
    public let dqnOptimizerConfig: DQNOptimizerConfig?
    public let tabularConfig: TabularConfig?

    public let explorationRate: Double?
    public let numGradientSteps: Int?

    public let targetEntropy: Float?
    public let entCoefConfig: EntropyCoef?

    public let nStates: Int?
    public let nActions: Int?
    public let stateStrides: [Int]?
    public let seed: UInt64?

    public init(
        version: String = "1.0",
        algorithmKind: AlgorithmKind,
        numTimesteps: Int,
        totalTimesteps: Int,
        currentProgressRemaining: Double,
        learningRateSchedule: LearningRateScheduleData?,
        offPolicyConfig: OffPolicyConfig? = nil,
        dqnConfig: DQNConfig? = nil,
        dqnPolicyConfig: DQNPolicyConfig? = nil,
        dqnOptimizerConfig: DQNOptimizerConfig? = nil,
        tabularConfig: TabularConfig? = nil,
        explorationRate: Double? = nil,
        numGradientSteps: Int? = nil,
        targetEntropy: Float? = nil,
        entCoefConfig: EntropyCoef? = nil,
        nStates: Int? = nil,
        nActions: Int? = nil,
        stateStrides: [Int]? = nil,
        seed: UInt64? = nil
    ) {
        self.version = version
        self.algorithmKind = algorithmKind
        self.numTimesteps = numTimesteps
        self.totalTimesteps = totalTimesteps
        self.currentProgressRemaining = currentProgressRemaining
        self.learningRateSchedule = learningRateSchedule
        self.offPolicyConfig = offPolicyConfig
        self.dqnConfig = dqnConfig
        self.dqnPolicyConfig = dqnPolicyConfig
        self.dqnOptimizerConfig = dqnOptimizerConfig
        self.tabularConfig = tabularConfig
        self.explorationRate = explorationRate
        self.numGradientSteps = numGradientSteps
        self.targetEntropy = targetEntropy
        self.entCoefConfig = entCoefConfig
        self.nStates = nStates
        self.nActions = nActions
        self.stateStrides = stateStrides
        self.seed = seed
    }

    static let metadataFilename = "metadata.json"

    public func write(to directory: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: directory.appendingPathComponent(Self.metadataFilename))
    }

    public static func read(from directory: URL) throws -> AlgorithmCheckpoint {
        let data = try Data(contentsOf: directory.appendingPathComponent(metadataFilename))
        return try JSONDecoder().decode(AlgorithmCheckpoint.self, from: data)
    }
}
