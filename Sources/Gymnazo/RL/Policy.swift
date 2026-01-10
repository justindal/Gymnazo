//
//  Policy.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Base Policy Protocol
///
/// Extends ``Model`` with action prediction capabilities.
///
/// - Parameters:
///     - squashOutput: For continuous actions, whether the output is squashed using tanh().
///     - featuresExtractor: The features extractor (required, not optional).
public protocol Policy: Model {
    var featuresExtractor: any FeaturesExtractor { get }
    var squashOutput: Bool { get }

    /// Get the action according to the policy for a given observation.
    ///
    /// - Parameters:
    ///   - observation: The preprocessed observation tensor.
    ///   - deterministic: Whether to use deterministic or stochastic actions.
    /// - Returns: The action tensor.
    func predictInternal(observation: MLXArray, deterministic: Bool) -> MLXArray
}

extension Policy {
    public var squashOutput: Bool { false }

    /// Get the policy action from an observation.
    ///
    /// - Parameters:
    ///   - observation: The input observation.
    ///   - deterministic: Whether to return deterministic actions.
    /// - Returns: The action to take.
    public func predict(observation: MLXArray, deterministic: Bool = false)
        -> MLXArray
    {
        setTrainingMode(false)

        let features = extractFeatures(
            obs: observation,
            featuresExtractor: featuresExtractor
        )

        var actions = predictInternal(
            observation: features,
            deterministic: deterministic
        )

        if let box = actionSpace as? Box {
            if squashOutput {
                actions = unscaleAction(actions)
            } else {
                actions = MLX.clip(actions, min: box.low, max: box.high)
            }
        }

        return actions
    }

    /// Get the policy action from a Dict observation.
    ///
    /// - Parameters:
    ///   - observation: The Dict observation.
    ///   - deterministic: Whether to return deterministic actions.
    /// - Returns: The action to take.
    public func predict(observation: [String: MLXArray], deterministic: Bool = false)
        -> MLXArray
    {
        setTrainingMode(false)

        guard let dictExtractor = featuresExtractor as? any DictFeaturesExtractor else {
            preconditionFailure(
                "predict(observation: [String: MLXArray]) requires a DictFeaturesExtractor")
        }

        let features = extractFeatures(
            obs: observation,
            featuresExtractor: dictExtractor
        )

        var actions = predictInternal(
            observation: features,
            deterministic: deterministic
        )

        if let box = actionSpace as? Box {
            if squashOutput {
                actions = unscaleAction(actions)
            } else {
                actions = MLX.clip(actions, min: box.low, max: box.high)
            }
        }

        return actions
    }

    /// Rescale the action from [low, high] to [-1, 1].
    ///
    /// - Parameter action: Action to scale.
    /// - Returns: Scaled action in [-1, 1].
    public func scaleAction(_ action: MLXArray) -> MLXArray {
        guard let box = actionSpace as? Box else {
            preconditionFailure("scaleAction requires a Box action space")
        }
        let low = box.low
        let high = box.high
        return 2.0 * ((action - low) / (high - low)) - 1.0
    }

    /// Rescale the action from [-1, 1] to [low, high].
    ///
    /// - Parameter scaledAction: Action in [-1, 1] to unscale.
    /// - Returns: Unscaled action in [low, high].
    public func unscaleAction(_ scaledAction: MLXArray) -> MLXArray {
        guard let box = actionSpace as? Box else {
            preconditionFailure("unscaleAction requires a Box action space")
        }
        let low = box.low
        let high = box.high
        return low + (0.5 * (scaledAction + 1.0) * (high - low))
    }
}

/// Orthogonal weight initialization (used in PPO and A2C).
///
/// Initializes Linear and Conv2d layers with orthogonal weights.
/// Uses QR decomposition to generate orthogonal matrices.
///
/// - Parameters:
///   - module: The module to initialize.
///   - gain: The scaling factor for the weights.
public func initWeightsOrthogonal(_ module: Module, gain: Float = 1.0) throws {
    if let linear = module as? Linear {
        let shape = linear.weight.shape
        let rows = shape[0]
        let cols = shape[1]

        let random = MLX.normal([max(rows, cols), max(rows, cols)])
        let (q, _) = MLXLinalg.qr(random, stream: .default)

        let orthogonal = q[0..<rows, 0..<cols] * gain

        var params: [String: MLXArray] = ["weight": orthogonal]
        if linear.bias != nil {
            params["bias"] = MLX.zeros([rows])
        }

        try linear.update(
            parameters: ModuleParameters.unflattened(params),
            verify: .none
        )

    } else if let conv = module as? Conv2d {
        let shape = conv.weight.shape
        let fanOut = shape[0]
        let fanIn = shape[1..<shape.count].reduce(1, *)

        let random = MLX.normal([max(fanOut, fanIn), max(fanOut, fanIn)])
        let (q, _) = MLXLinalg.qr(random, stream: .default)

        let orthogonal = q[0..<fanOut, 0..<fanIn].reshaped(shape) * gain

        var params: [String: MLXArray] = ["weight": orthogonal]
        if conv.bias != nil {
            params["bias"] = MLX.zeros([shape[0]])
        }
        try conv.update(
            parameters: ModuleParameters.unflattened(params),
            verify: .none
        )
    }
}
