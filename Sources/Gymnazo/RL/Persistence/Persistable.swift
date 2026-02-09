//
//  Persistable.swift
//  Gymnazo
//

import Foundation
import MLX
import MLXNN

public enum PersistenceError: Error, LocalizedError {
    case missingFile(String)
    case invalidCheckpoint(String)
    case incompatibleVersion(String)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let name):
            return "Missing required file: \(name)"
        case .invalidCheckpoint(let reason):
            return "Invalid checkpoint: \(reason)"
        case .incompatibleVersion(let version):
            return "Incompatible checkpoint version: \(version)"
        }
    }
}

enum CheckpointFiles {
    static let policy = "policy.safetensors"
    static let target = "target.safetensors"
    static let critic = "critic.safetensors"
    static let criticTarget = "critic_target.safetensors"
    static let entropy = "entropy.safetensors"
    static let qTable = "q_table.safetensors"
    static let bufferDirectory = "replay_buffer"
}

extension Module {
    func saveWeights(to url: URL) throws {
        let flattened = self.parameters().flattened()
        let weights = Dictionary(uniqueKeysWithValues: flattened)
        eval(weights.values)
        try MLX.save(arrays: weights, url: url)
    }

    func loadWeights(from url: URL) throws {
        let loaded = try MLX.loadArrays(url: url)
        try self.update(parameters: ModuleParameters.unflattened(loaded), verify: .noUnusedKeys)
        eval(self.parameters())
    }
}
