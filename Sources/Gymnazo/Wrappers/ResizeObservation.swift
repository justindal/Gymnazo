//
//  ResizeObservation.swift
//  Gymnazo
//
//  Resizes image observations to a specified shape.
//

import MLX

/// Resizes image observations to a specified shape.
///
/// This wrapper resizes image observations using nearest-neighbor interpolation,
/// allowing you to standardize observation dimensions across different environments.
///
/// ## Overview
///
/// `ResizeObservation` is commonly used to:
/// - Match observations to standard CNN input sizes (e.g., 84×84 for Atari-style networks)
/// - Reduce observation dimensionality for faster training
/// - Standardize inputs when combining different environments
///
/// ## Example
///
/// ```swift
/// var env = CarRacing()
///     .resized(to: (84, 84))     // [96, 96, 3] -> [84, 84, 3]
///     .frameStacked(4)
///     .timeLimited(1000)
/// ```
///
/// ## Topics
///
/// ### Creating a Resize Wrapper
/// - ``init(env:shape:)``
///
/// ### Configuration
/// - ``targetHeight``
/// - ``targetWidth``
public struct ResizeObservation: Wrapper {
    public var env: any Env
    public let targetHeight: Int
    public let targetWidth: Int
    public let observationSpace: any Space
    private let sourceHeight: Int
    private let sourceWidth: Int
    private let indicesH: MLXArray
    private let indicesW: MLXArray

    public var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }
    public var renderMode: RenderMode? {
        get { env.renderMode }
        set { env.renderMode = newValue }
    }

    /// Creates a resize observation wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap (must have MLXArray observations with shape [H, W, ...])
    ///   - shape: Target (height, width) for the resized observations
    public init(env: any Env, shape: (Int, Int)) throws {
        self.env = env
        self.targetHeight = shape.0
        self.targetWidth = shape.1

        guard let innerBox = env.observationSpace as? Box,
            let innerShape = innerBox.shape,
            innerShape.count >= 2
        else {
            throw GymnazoError.invalidResizeShape
        }
        let sourceHeight = innerShape[0]
        let sourceWidth = innerShape[1]
        self.sourceHeight = sourceHeight
        self.sourceWidth = sourceWidth
        self.indicesH = Self.resizeIndices(source: sourceHeight, target: targetHeight)
        self.indicesW = Self.resizeIndices(source: sourceWidth, target: targetWidth)

        var newShape = [targetHeight, targetWidth]
        if innerShape.count > 2 {
            newShape.append(contentsOf: innerShape[2...])
        }

        self.observationSpace = Box(
            low: 0,
            high: 255,
            shape: newShape,
            dtype: innerBox.dtype ?? .uint8
        )
    }

    public mutating func step(_ action: MLXArray) throws -> Step {
        let result = try env.step(action)
        return Step(
            obs: resize(result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        let result = try env.reset(seed: seed, options: options)
        return Reset(obs: resize(result.obs), info: result.info)
    }

    private func resize(_ obs: MLXArray) -> MLXArray {
        let originalShape = obs.shape

        let inputNdim = originalShape.count

        var input = obs.asType(.float32)

        if inputNdim == 2 {
            input = input.expandedDimensions(axis: 0).expandedDimensions(axis: -1)
        } else if inputNdim == 3 {
            input = input.expandedDimensions(axis: 0)
        }

        let indicesHArray: MLXArray
        let indicesWArray: MLXArray
        if originalShape[0] == sourceHeight && originalShape[1] == sourceWidth {
            indicesHArray = indicesH
            indicesWArray = indicesW
        } else {
            indicesHArray = Self.resizeIndices(source: originalShape[0], target: targetHeight)
            indicesWArray = Self.resizeIndices(source: originalShape[1], target: targetWidth)
        }

        var resized = input.take(indicesHArray, axis: 1)
        resized = resized.take(indicesWArray, axis: 2)

        resized = resized.squeezed(axis: 0)

        if inputNdim == 2 {
            resized = resized.squeezed(axis: -1)
        }

        return resized.asType(observationSpace.dtype ?? .uint8)
    }

    private static func resizeIndices(source: Int, target: Int) -> MLXArray {
        let scale = Float(source) / Float(target)
        var indices = [Int32]()
        indices.reserveCapacity(target)
        for i in 0..<target {
            let src = min(Int(Float(i) * scale), source - 1)
            indices.append(Int32(src))
        }
        return MLXArray(indices)
    }
}
