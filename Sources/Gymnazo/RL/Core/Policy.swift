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

    func callAsFunction(_ observation: MLXArray, deterministic: Bool) -> MLXArray
    func predict(observation: MLXArray, deterministic: Bool) -> MLXArray
    func predict(observation: [String: MLXArray], deterministic: Bool) -> MLXArray
}

extension Policy {
    public var squashOutput: Bool { false }

    public func predict(
        observation: MLXArray,
        deterministic: Bool = false
    ) -> MLXArray {
        setTrainingMode(false)

        let features = extractFeatures(
            obs: observation,
            featuresExtractor: featuresExtractor
        )

        var actions = self(
            features,
            deterministic: deterministic
        )

        if let box = boxSpace(from: actionSpace) {
            if squashOutput {
                let low = box.low
                let high = box.high
                let unscaled = low + (0.5 * (actions + 1.0) * (high - low))
                actions = MLX.clip(unscaled, min: low, max: high)
            } else {
                actions = MLX.clip(actions, min: box.low, max: box.high)
            }
        }

        return actions
    }

    public func predict(
        observation: [String: MLXArray],
        deterministic: Bool = false
    ) -> MLXArray {
        setTrainingMode(false)

        guard let dictExtractor = featuresExtractor as? any DictFeaturesExtractor else {
            preconditionFailure(
                "predict(observation: [String: MLXArray]) requires a DictFeaturesExtractor")
        }

        let features = extractFeatures(
            obs: observation,
            featuresExtractor: dictExtractor
        )

        var actions = self(
            features,
            deterministic: deterministic
        )

        if let box = boxSpace(from: actionSpace) {
            if squashOutput {
                let low = box.low
                let high = box.high
                let unscaled = low + (0.5 * (actions + 1.0) * (high - low))
                actions = MLX.clip(unscaled, min: low, max: high)
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
    public func scaleAction(_ action: MLXArray) throws -> MLXArray {
        guard let box = boxSpace(from: actionSpace) else {
            throw GymnazoError.invalidPolicyActionSpace(
                operation: "scaleAction",
                expected: "Box",
                actual: String(describing: type(of: actionSpace))
            )
        }
        let low = box.low
        let high = box.high
        let scaled = 2.0 * ((action - low) / (high - low)) - 1.0
        return MLX.clip(scaled, min: -1.0, max: 1.0)
    }

    /// Rescale the action from [-1, 1] to [low, high].
    ///
    /// - Parameter scaledAction: Action in [-1, 1] to unscale.
    /// - Returns: Unscaled action in [low, high].
    public func unscaleAction(_ scaledAction: MLXArray) throws -> MLXArray {
        guard let box = boxSpace(from: actionSpace) else {
            throw GymnazoError.invalidPolicyActionSpace(
                operation: "unscaleAction",
                expected: "Box",
                actual: String(describing: type(of: actionSpace))
            )
        }
        let low = box.low
        let high = box.high
        let unscaled = low + (0.5 * (scaledAction + 1.0) * (high - low))
        var clipped = MLX.clip(unscaled, min: low, max: high)
        if let shape = box.shape, clipped.size == shape.reduce(1, *) {
            clipped = clipped.reshaped(shape)
        }
        return clipped
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

        let random = MLX.normal([max(rows, cols), max(rows, cols)], stream: .cpu)
        let (q, _) = MLXLinalg.qr(random, stream: .cpu)

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

        let random = MLX.normal([max(fanOut, fanIn), max(fanOut, fanIn)], stream: .cpu)
        let (q, _) = MLXLinalg.qr(random, stream: .cpu)

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
