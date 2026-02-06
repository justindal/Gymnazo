//
//  EvalCallback.swift
//  Gymnazo
//

import Foundation
import MLX

/// Configuration for evaluation callback.
public struct EvalCallbackConfig: Sendable {
    public let evalFreq: Int
    public let nEvalEpisodes: Int
    public let deterministic: Bool
    public let render: Bool
    public let verbose: Int

    /// Creates an evaluation callback configuration.
    ///
    /// - Parameters:
    ///   - evalFreq: Evaluate every N timesteps.
    ///   - nEvalEpisodes: Number of episodes per evaluation.
    ///   - deterministic: Use deterministic actions during evaluation.
    ///   - render: Render the environment during evaluation.
    ///   - verbose: Verbosity level (0=silent, 1=info).
    public init(
        evalFreq: Int = 10_000,
        nEvalEpisodes: Int = 5,
        deterministic: Bool = true,
        render: Bool = false,
        verbose: Int = 1
    ) {
        self.evalFreq = evalFreq
        self.nEvalEpisodes = nEvalEpisodes
        self.deterministic = deterministic
        self.render = render
        self.verbose = verbose
    }
}

/// Callback for evaluating the policy periodically during training.
///
/// Evaluates the current policy on a separate evaluation environment and optionally
/// saves the best model based on mean reward.
public final class EvalCallback<P: Policy>: Callback {
    public let config: EvalCallbackConfig
    public let bestModelSavePath: URL?
    public var logger: (any Logger)?

    private var evalEnv: any Env
    private let policy: P
    private var bestMeanReward: Double = -.infinity
    private var lastEvalTimestep: Int = 0
    private var evalCount: Int = 0

    /// The best mean reward achieved during evaluation.
    public var bestReward: Double { bestMeanReward }

    /// Number of evaluations performed.
    public var evaluationCount: Int { evalCount }

    /// Creates an evaluation callback.
    ///
    /// - Parameters:
    ///   - evalEnv: Environment for evaluation (separate from training).
    ///   - policy: The policy to evaluate.
    ///   - config: Evaluation configuration.
    ///   - bestModelSavePath: Path to save the best model (optional).
    ///   - logger: Logger for recording evaluation metrics.
    public init(
        evalEnv: any Env,
        policy: P,
        config: EvalCallbackConfig = EvalCallbackConfig(),
        bestModelSavePath: URL? = nil,
        logger: (any Logger)? = nil
    ) {
        self.evalEnv = evalEnv
        self.policy = policy
        self.config = config
        self.bestModelSavePath = bestModelSavePath
        self.logger = logger
    }

    public func onStep(locals: CallbackLocals) -> Bool {
        if locals.numTimesteps - lastEvalTimestep < config.evalFreq {
            return true
        }

        lastEvalTimestep = locals.numTimesteps
        evalCount += 1

        do {
            let result = try evaluatePolicy(
                policy: policy,
                env: &evalEnv,
                nEvalEpisodes: config.nEvalEpisodes,
                deterministic: config.deterministic,
                render: config.render
            )

            logger?.record(LogKey.Eval.meanReward, value: result.meanReward)
            logger?.record(LogKey.Eval.stdReward, value: result.stdReward)
            logger?.record(LogKey.Eval.meanEpLength, value: result.meanEpisodeLength)

            if config.verbose >= 1 {
                print(
                    "Eval \(evalCount): mean_reward=\(String(format: "%.2f", result.meanReward)) +/- \(String(format: "%.2f", result.stdReward))"
                )
            }

            if result.meanReward > bestMeanReward {
                bestMeanReward = result.meanReward

                if let savePath = bestModelSavePath {
                    if config.verbose >= 1 {
                        print(
                            "New best mean reward: \(String(format: "%.2f", bestMeanReward)). Saving model..."
                        )
                    }
                    try policy.save(to: savePath)
                }
            }
        } catch {
            if config.verbose >= 1 {
                print("Evaluation failed: \(error)")
            }
        }

        return true
    }
}
