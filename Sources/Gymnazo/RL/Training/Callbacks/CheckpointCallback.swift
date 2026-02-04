//
//  CheckpointCallback.swift
//  Gymnazo
//

import Foundation

/// Callback for saving model checkpoints at regular intervals.
public final class CheckpointCallback<P: Policy>: Callback {
    public let saveFreq: Int
    public let savePath: URL
    public let namePrefix: String
    public let verbose: Int

    private let policy: P
    private var lastSaveTimestep: Int = 0
    private var checkpointCount: Int = 0

    /// Number of checkpoints saved.
    public var savedCount: Int { checkpointCount }

    /// Creates a checkpoint callback.
    ///
    /// - Parameters:
    ///   - policy: The policy to save.
    ///   - saveFreq: Save every N timesteps.
    ///   - savePath: Directory to save checkpoints.
    ///   - namePrefix: Prefix for checkpoint filenames.
    ///   - verbose: Verbosity level (0=silent, 1=info).
    public init(
        policy: P,
        saveFreq: Int,
        savePath: URL,
        namePrefix: String = "model",
        verbose: Int = 1
    ) {
        self.policy = policy
        self.saveFreq = saveFreq
        self.savePath = savePath
        self.namePrefix = namePrefix
        self.verbose = verbose
    }

    public func onTrainingStart(locals: CallbackLocals) {
        do {
            try FileManager.default.createDirectory(at: savePath, withIntermediateDirectories: true)
        } catch {
            if verbose >= 1 {
                print("Failed to create checkpoint directory: \(error)")
            }
        }
    }

    public func onStep(locals: CallbackLocals) -> Bool {
        if locals.numTimesteps - lastSaveTimestep < saveFreq {
            return true
        }

        lastSaveTimestep = locals.numTimesteps
        checkpointCount += 1

        let filename = "\(namePrefix)_\(locals.numTimesteps)_steps.safetensors"
        let fileURL = savePath.appendingPathComponent(filename)

        do {
            try policy.save(to: fileURL)
            if verbose >= 1 {
                print("Saved checkpoint: \(filename)")
            }
        } catch {
            if verbose >= 1 {
                print("Failed to save checkpoint: \(error)")
            }
        }

        return true
    }
}

/// Callback for saving the model only when it improves.
public final class BestModelCallback: Callback {
    public let savePath: URL
    public let verbose: Int

    private var bestValue: Double = -.infinity
    private let valueProvider: () -> Double?
    private let saveAction: (URL) throws -> Void

    /// Creates a best model callback.
    ///
    /// - Parameters:
    ///   - savePath: Path to save the best model.
    ///   - valueProvider: Closure that returns the current metric value.
    ///   - saveAction: Closure that performs the save operation.
    ///   - verbose: Verbosity level.
    public init(
        savePath: URL,
        valueProvider: @escaping () -> Double?,
        saveAction: @escaping (URL) throws -> Void,
        verbose: Int = 1
    ) {
        self.savePath = savePath
        self.valueProvider = valueProvider
        self.saveAction = saveAction
        self.verbose = verbose
    }

    public func onStep(locals: CallbackLocals) -> Bool {
        guard let currentValue = valueProvider() else { return true }

        if currentValue > bestValue {
            bestValue = currentValue

            do {
                try saveAction(savePath)
                if verbose >= 1 {
                    print("New best value: \(String(format: "%.4f", bestValue)). Model saved.")
                }
            } catch {
                if verbose >= 1 {
                    print("Failed to save best model: \(error)")
                }
            }
        }

        return true
    }

    /// The current best value.
    public var best: Double { bestValue }
}
