//
//  TabularAgent+Persistence.swift
//  Gymnazo
//

import Foundation
import MLX

extension TabularAgent {
    public func save(to directory: URL) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let table = self.table
        eval(table)

        try MLX.save(
            arrays: ["q_table": table],
            url: directory.appendingPathComponent(CheckpointFiles.qTable)
        )

        let kind: AlgorithmKind = updateRule == .sarsa ? .sarsa : .qLearning

        let checkpoint = AlgorithmCheckpoint(
            algorithmKind: kind,
            numTimesteps: numTimesteps,
            totalTimesteps: 0,
            currentProgressRemaining: 0,
            learningRateSchedule: nil,
            tabularConfig: config,
            explorationRate: epsilon,
            nStates: numStates,
            nActions: numActions,
            stateStrides: strides.isEmpty ? nil : strides
        )
        try checkpoint.write(to: directory)
    }

    public static func load(
        from directory: URL,
        env: (any Env)? = nil
    ) throws -> TabularAgent {
        let checkpoint = try AlgorithmCheckpoint.read(from: directory)

        guard checkpoint.algorithmKind == .qLearning || checkpoint.algorithmKind == .sarsa else {
            throw PersistenceError.invalidCheckpoint(
                "Expected tabular checkpoint, got \(checkpoint.algorithmKind)")
        }

        guard let savedConfig = checkpoint.tabularConfig else {
            throw PersistenceError.invalidCheckpoint("Missing tabular config")
        }

        guard let nStates = checkpoint.nStates, let nActions = checkpoint.nActions else {
            throw PersistenceError.invalidCheckpoint("Missing nStates or nActions")
        }

        let updateRule: UpdateRule = checkpoint.algorithmKind == .sarsa ? .sarsa : .qLearning
        let explorationRate = checkpoint.explorationRate ?? savedConfig.epsilon

        let loaded = try MLX.loadArrays(
            url: directory.appendingPathComponent(CheckpointFiles.qTable))
        let qTable: MLXArray
        if let table = loaded["q_table"] {
            eval(table)
            qTable = table
        } else {
            qTable = MLX.zeros([nStates, nActions])
        }

        let savedStrides: [Int]
        if let s = checkpoint.stateStrides, !s.isEmpty {
            savedStrides = s
        } else if let env {
            savedStrides = TabularAgent.stateSpaceInfo(from: env.observationSpace).strides
        } else {
            savedStrides = []
        }

        let agent = TabularAgent(
            updateRule: updateRule,
            config: savedConfig,
            numStates: nStates,
            numActions: nActions,
            seed: checkpoint.seed,
            stateStrides: savedStrides,
            qTable: qTable,
            timesteps: checkpoint.numTimesteps,
            explorationRate: explorationRate
        )

        if let env {
            agent.setEnv(env)
        }

        return agent
    }
}
