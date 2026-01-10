//
//  StateDependentNoiseDistribution.swift
//  Gymnazo
//

import MLX
import MLXNN

/// State-Dependent Exploration (gSDE) distribution.
///
/// Implements generalized State-Dependent Exploration as described in
/// "Smooth Exploration for Robotic Reinforcement Learning" (Raffin et al., 2021).
///
/// The exploration noise depends on the current state features, providing
/// smoother and more consistent exploration compared to action-space noise.
public final class StateDependentNoiseDistribution: Distribution {
    private let actionDim: Int
    private let latentSDEDim: Int
    private let fullStd: Bool
    private let useExpln: Bool
    private let squashOutput: Bool
    private let learnFeatures: Bool

    private var mean: MLXArray
    private var logStd: MLXArray
    private var latentSDE: MLXArray
    private var explorationMatrix: MLXArray

    private let epsilon: Float = 1e-6

    /// Creates a StateDependentNoiseDistribution.
    ///
    /// - Parameters:
    ///   - actionDim: Dimension of the action space.
    ///   - fullStd: If true, use (latent_dim x action_dim) parameters for std.
    ///   - useExpln: If true, use expln() instead of exp() for positive std.
    ///   - squashOutput: If true, squash output using tanh.
    ///   - learnFeatures: If true, learn features for exploration.
    ///   - latentSDEDim: Dimension of the latent SDE features (defaults to actionDim).
    public init(
        actionDim: Int,
        fullStd: Bool = true,
        useExpln: Bool = false,
        squashOutput: Bool = false,
        learnFeatures: Bool = false,
        latentSDEDim: Int? = nil
    ) {
        self.actionDim = actionDim
        self.latentSDEDim = latentSDEDim ?? actionDim
        self.fullStd = fullStd
        self.useExpln = useExpln
        self.squashOutput = squashOutput
        self.learnFeatures = learnFeatures

        self.mean = MLXArray([])
        self.logStd = MLXArray([])
        self.latentSDE = MLXArray([])
        self.explorationMatrix = MLXArray([])
    }

    /// Creates the network for producing mean actions and log_std.
    ///
    /// - Parameters:
    ///   - latentDim: Dimension of the latent features.
    ///   - actionDim: Dimension of the action space.
    ///   - logStdInit: Initial value for log standard deviation.
    ///   - fullStd: If true, use (latent_dim x action_dim) parameters for std.
    /// - Returns: Action network and trainable log_std parameter.
    public static func probaDistributionNet(
        latentDim: Int,
        actionDim: Int,
        logStdInit: Float = -2.0,
        fullStd: Bool = true
    ) -> (Linear, MLXArray) {
        let meanActionsNet = Linear(latentDim, actionDim)

        let logStdShape: [Int]
        if fullStd {
            logStdShape = [latentDim, actionDim]
        } else {
            logStdShape = [latentDim, 1]
        }

        let logStd = MLX.zeros(logStdShape) + logStdInit

        return (meanActionsNet, logStd)
    }

    /// Sets the distribution parameters.
    ///
    /// - Parameters:
    ///   - meanActions: Mean of the distribution.
    ///   - logStd: Log standard deviation parameter.
    ///   - latentSDE: Latent features for state-dependent exploration.
    /// - Returns: Self for chaining.
    @discardableResult
    public func probaDistribution(
        meanActions: MLXArray,
        logStd: MLXArray,
        latentSDE: MLXArray
    ) -> Self {
        let logStdShape = logStd.shape
        precondition(
            logStdShape.count == 2 && logStdShape[0] == latentSDEDim,
            "Expected logStd shape [\(latentSDEDim), *], got \(logStdShape)."
        )
        if fullStd {
            precondition(
                logStdShape[1] == actionDim,
                "Expected logStd shape [\(latentSDEDim), \(actionDim)] (fullStd=true), got \(logStdShape)."
            )
        } else {
            precondition(
                logStdShape[1] == 1,
                "Expected logStd shape [\(latentSDEDim), 1] (fullStd=false), got \(logStdShape)."
            )
        }

        self.mean = meanActions
        self.logStd = logStd
        self.latentSDE = learnFeatures ? latentSDE : MLX.stopGradient(latentSDE)
        return self
    }

    private func sampleStandardNormal(_ shape: [Int], key: MLXArray?) -> MLXArray {
        if let key = key {
            return MLX.normal(shape, key: key)
        }
        return MLX.normal(shape)
    }

    /// Samples new exploration weights.
    ///
    /// - Parameters:
    ///   - batchSize: Number of environments.
    ///   - key: Optional RNG key for reproducible sampling.
    public func sampleWeights(batchSize: Int = 1, key: MLXArray? = nil) {
        sampleWeights(logStd: logStd, batchSize: batchSize, key: key)
    }

    /// Gets the standard deviation from log_std.
    public func getStd(logStd: MLXArray) -> MLXArray {
        getStdInternal(logStd)
    }

    /// Samples new exploration weights with explicit logStd.
    ///
    /// - Parameters:
    ///   - logStd: Log standard deviation parameter.
    ///   - batchSize: Number of environments.
    ///   - key: Optional RNG key for reproducible sampling.
    public func sampleWeights(logStd: MLXArray, batchSize: Int = 1, key: MLXArray? = nil) {
        let std = getStdInternal(logStd)
        explorationMatrix =
            sampleStandardNormal([batchSize, latentSDEDim, actionDim], key: key) * std
    }

    private func getStd() -> MLXArray {
        getStdInternal(logStd)
    }

    private func getStdInternal(_ logStd: MLXArray) -> MLXArray {
        if useExpln {
            return expln(logStd)
        }
        return MLX.exp(logStd)
    }

    /// Exponential linear unit for ensuring positive std.
    ///
    /// expln(x) = exp(x) if x <= 0 else x + 1
    /// This prevents std from growing too fast while keeping it positive.
    private func expln(_ x: MLXArray) -> MLXArray {
        let belowZero = MLX.exp(x)
        let aboveZero = x + 1.0
        return MLX.where(x .<= 0, belowZero, aboveZero)
    }

    /// Computes exploration noise from latent features.
    ///
    /// - Parameter key: Key for reproducible sampling if weights need initialization.
    private func getExplorationNoise(key: MLXArray? = nil) -> MLXArray {
        let latentShape = latentSDE.shape
        let batchSize = latentShape.count > 1 ? latentShape[0] : 1

        if explorationMatrix.size == 0 {
            sampleWeights(batchSize: batchSize, key: key)
        }

        let latent = latentSDE.reshaped([batchSize, 1, latentSDEDim])
        let noise = MLX.matmul(latent, explorationMatrix).squeezed(axis: 1)
        return noise
    }

    public func logProb(_ actions: MLXArray) -> MLXArray {
        var actionsForLogProb = actions

        if squashOutput {
            let gaussianActions = inverseSquash(actions)
            actionsForLogProb = gaussianActions
        }

        let std = getStd()
        let variance = MLX.matmul(latentSDE * latentSDE, std * std)

        let diff = actionsForLogProb - mean
        let logP =
            -0.5
            * (diff * diff / (variance + epsilon) + MLX.log(variance + epsilon)
                + Float.log(2.0 * Float.pi))
        var logProb = MLX.sum(logP, axis: -1)

        if squashOutput {
            logProb = logProb - MLX.sum(MLX.log(1.0 - actions * actions + epsilon), axis: -1)
        }

        return logProb
    }

    public func entropy() -> MLXArray? {
        if squashOutput {
            return nil
        }

        let std = getStd()
        let variance = MLX.matmul(latentSDE * latentSDE, std * std)
        let entropy = 0.5 * MLX.sum(MLX.log(2.0 * Float.pi * (variance + epsilon)) + 1.0, axis: -1)
        return entropy
    }

    public func sample(key: MLXArray? = nil) -> MLXArray {
        let noise = getExplorationNoise(key: key)
        var actions = mean + noise

        if squashOutput {
            actions = MLX.tanh(actions)
        }

        return actions
    }

    public func mode() -> MLXArray {
        var actions = mean

        if squashOutput {
            actions = MLX.tanh(actions)
        }

        return actions
    }

    /// Inverse of the squashing function (tanh).
    private func inverseSquash(_ actions: MLXArray) -> MLXArray {
        let clipped = MLX.clip(actions, min: MLXArray(-1.0 + epsilon), max: MLXArray(1.0 - epsilon))
        return 0.5 * (MLX.log(1.0 + clipped) - MLX.log(1.0 - clipped))
    }
}
