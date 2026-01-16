//
//  GrayscaleObservation.swift
//  Gymnazo
//
//  Converts RGB image observations to grayscale.
//

import MLX

/// Converts RGB image observations to grayscale.
///
/// This wrapper reduces the dimensionality of image observations by converting
/// 3-channel RGB images to single-channel grayscale using standard luminance weights.
///
/// ## Overview
///
/// `GrayscaleObservation` uses the ITU-R BT.601 standard for luminance:
/// ```
/// grayscale = 0.299*R + 0.587*G + 0.114*B
/// ```
///
/// This is commonly used in reinforcement learning to reduce the observation space
/// complexity while preserving important visual information.
///
/// ## Example
///
/// ```swift
/// var env = CarRacing()
///     .grayscale()           // [96, 96, 3] -> [96, 96]
///     .timeLimited(1000)
///
/// // Or with keepDim to preserve channel dimension:
/// var env = CarRacing()
///     .grayscale(keepDim: true)  // [96, 96, 3] -> [96, 96, 1]
/// ```
///
/// ## Topics
///
/// ### Creating a Grayscale Wrapper
/// - ``init(env:keepDim:)``
///
/// ### Configuration
/// - ``keepDim``
public struct GrayscaleObservation<BaseEnv: Env>: Env
where BaseEnv.Observation == MLXArray {
    public var env: BaseEnv
    public let keepDim: Bool
    public let observationSpace: Box

    public var actionSpace: BaseEnv.ActionSpace { env.actionSpace }
    public var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }
    public var renderMode: String? {
        get { env.renderMode }
        set { env.renderMode = newValue }
    }

    /// Creates a grayscale observation wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap (must have MLXArray observations with shape [H, W, 3])
    ///   - keepDim: If true, output shape is [H, W, 1]. If false, output shape is [H, W].
    public init(env: BaseEnv, keepDim: Bool = false) {
        self.env = env
        self.keepDim = keepDim

        guard let innerBox = env.observationSpace as? Box,
            let shape = innerBox.shape,
            shape.count == 3,
            shape[2] == 3
        else {
            fatalError("GrayscaleObservation requires Box observation space with shape [H, W, 3]")
        }

        let newShape: [Int] = keepDim ? [shape[0], shape[1], 1] : [shape[0], shape[1]]
        self.observationSpace = Box(
            low: 0,
            high: 255,
            shape: newShape,
            dtype: innerBox.dtype ?? .uint8
        )
    }

    public mutating func step(_ action: BaseEnv.Action) -> Step<MLXArray> {
        let result = env.step(action)
        return Step(
            obs: toGrayscale(result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<MLXArray> {
        let result = env.reset(seed: seed, options: options)
        return Reset(obs: toGrayscale(result.obs), info: result.info)
    }

    public var unwrapped: any Env { env.unwrapped }

    @discardableResult
    public func render() -> Any? { env.render() }

    public func close() { env.close() }

    private func toGrayscale(_ obs: MLXArray) -> MLXArray {
        // ITU-R BT.601 standard weights
        let r = obs[.ellipsis, 0].asType(.float32)
        let g = obs[.ellipsis, 1].asType(.float32)
        let b = obs[.ellipsis, 2].asType(.float32)

        let gray = 0.299 * r + 0.587 * g + 0.114 * b

        let result: MLXArray
        if keepDim {
            result = gray.expandedDimensions(axis: -1)
        } else {
            result = gray
        }

        return result.asType(observationSpace.dtype ?? .uint8)
    }
}
