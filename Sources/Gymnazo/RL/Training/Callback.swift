//
//  Callback.swift
//  Gymnazo
//

import Foundation

/// Event types that can trigger callback invocations.
public enum CallbackEvent: Sendable {
    case trainingStart
    case step
    case rolloutStart
    case rolloutEnd
    case trainingEnd
}

/// Locals passed to callbacks containing current training state.
public struct CallbackLocals: Sendable {
    public let numTimesteps: Int
    public let totalTimesteps: Int
    public let numEpisodes: Int
    public let iteration: Int

    public init(
        numTimesteps: Int = 0,
        totalTimesteps: Int = 0,
        numEpisodes: Int = 0,
        iteration: Int = 0
    ) {
        self.numTimesteps = numTimesteps
        self.totalTimesteps = totalTimesteps
        self.numEpisodes = numEpisodes
        self.iteration = iteration
    }
}

/// Protocol for training callbacks that hook into the learning loop.
///
/// Callbacks provide hooks at various points during training to monitor progress,
/// save checkpoints, implement early stopping, or log metrics.
public protocol Callback: AnyObject {
    /// Called before the first rollout starts.
    ///
    /// - Parameters:
    ///   - locals: Current training state.
    func onTrainingStart(locals: CallbackLocals)

    /// Called after each environment step.
    ///
    /// - Parameters:
    ///   - locals: Current training state.
    /// - Returns: `true` to continue training, `false` to stop early.
    func onStep(locals: CallbackLocals) -> Bool

    /// Called at the start of a rollout collection phase.
    ///
    /// - Parameters:
    ///   - locals: Current training state.
    func onRolloutStart(locals: CallbackLocals)

    /// Called at the end of a rollout, before the policy update.
    ///
    /// - Parameters:
    ///   - locals: Current training state.
    func onRolloutEnd(locals: CallbackLocals)

    /// Called when training ends.
    ///
    /// - Parameters:
    ///   - locals: Current training state.
    func onTrainingEnd(locals: CallbackLocals)
}

extension Callback {
    public func onTrainingStart(locals: CallbackLocals) {}
    public func onStep(locals: CallbackLocals) -> Bool { true }
    public func onRolloutStart(locals: CallbackLocals) {}
    public func onRolloutEnd(locals: CallbackLocals) {}
    public func onTrainingEnd(locals: CallbackLocals) {}
}

/// A callback that combines multiple callbacks into one.
///
/// All callbacks are invoked in order. For `onStep`, returns `false` if any callback returns `false`.
public final class CallbackList: Callback {
    public var callbacks: [any Callback]

    /// Creates a callback list from an array of callbacks.
    ///
    /// - Parameter callbacks: The callbacks to combine.
    public init(_ callbacks: [any Callback] = []) {
        self.callbacks = callbacks
    }

    /// Creates a callback list from variadic callbacks.
    ///
    /// - Parameter callbacks: The callbacks to combine.
    public convenience init(_ callbacks: any Callback...) {
        self.init(callbacks)
    }

    public func onTrainingStart(locals: CallbackLocals) {
        for callback in callbacks {
            callback.onTrainingStart(locals: locals)
        }
    }

    public func onStep(locals: CallbackLocals) -> Bool {
        var shouldContinue = true
        for callback in callbacks {
            if !callback.onStep(locals: locals) {
                shouldContinue = false
            }
        }
        return shouldContinue
    }

    public func onRolloutStart(locals: CallbackLocals) {
        for callback in callbacks {
            callback.onRolloutStart(locals: locals)
        }
    }

    public func onRolloutEnd(locals: CallbackLocals) {
        for callback in callbacks {
            callback.onRolloutEnd(locals: locals)
        }
    }

    public func onTrainingEnd(locals: CallbackLocals) {
        for callback in callbacks {
            callback.onTrainingEnd(locals: locals)
        }
    }
}

/// A callback that invokes a closure on each step.
public final class FunctionCallback: Callback {
    private let stepHandler: (CallbackLocals) -> Bool

    /// Creates a callback from a step handler closure.
    ///
    /// - Parameter onStep: Closure called on each step, returning whether to continue.
    public init(onStep: @escaping (CallbackLocals) -> Bool) {
        self.stepHandler = onStep
    }

    public func onStep(locals: CallbackLocals) -> Bool {
        stepHandler(locals)
    }
}
