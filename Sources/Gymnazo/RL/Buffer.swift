//
//  Buffer.swift
//  Gymnazo
//

import Foundation
import MLX

/// Base protocol for buffers (rollout or replay).
///
/// - Parameters:
///     - bufferSize: Max capacity of the buffer.
///     - observationSpace: Observation space of the environment.
///     - actionSpace: Action space of the environment.
///     - numEnvs: Number of parallel environments.
public protocol Buffer {
    var bufferSize: Int { get }
    var observationSpace: any Space<MLXArray> { get }
    var actionSpace: any Space { get }
    var numEnvs: Int { get }
    var count: Int { get }

    /// Reset the buffer.
    mutating func reset()
}

extension Buffer {
    public var isEmpty: Bool { count == 0 }
    public var isFull: Bool { count >= bufferSize }
}

/// Persistence support for buffers.
public protocol BufferPersisting: Buffer {
    func save(to url: URL) throws
    mutating func load(from url: URL) throws
}

/// A single rollout step.
public struct RolloutStep {
    public let observation: MLXArray
    public let action: MLXArray
    public let reward: MLXArray
    public let episodeStart: MLXArray
    public let value: MLXArray
    public let logProb: MLXArray

    public init(
        observation: MLXArray,
        action: MLXArray,
        reward: MLXArray,
        episodeStart: MLXArray,
        value: MLXArray,
        logProb: MLXArray
    ) {
        self.observation = observation
        self.action = action
        self.reward = reward
        self.episodeStart = episodeStart
        self.value = value
        self.logProb = logProb
    }
}

/// A sampled rollout batch.
public struct RolloutBatch {
    public let observations: MLXArray
    public let actions: MLXArray
    public let values: MLXArray
    public let logProbs: MLXArray
    public let advantages: MLXArray
    public let returns: MLXArray

    public init(
        observations: MLXArray,
        actions: MLXArray,
        values: MLXArray,
        logProbs: MLXArray,
        advantages: MLXArray,
        returns: MLXArray
    ) {
        self.observations = observations
        self.actions = actions
        self.values = values
        self.logProbs = logProbs
        self.advantages = advantages
        self.returns = returns
    }
}

/// Rollout buffer protocol for on-policy algorithms.
public protocol RolloutBuffer: Buffer {
    mutating func append(_ step: RolloutStep)

    mutating func computeReturnsAndAdvantages(
        lastValues: MLXArray,
        dones: MLXArray,
        gamma: Double,
        gaeLambda: Double
    )

    func batches(batchSize: Int) -> [RolloutBatch]
}
