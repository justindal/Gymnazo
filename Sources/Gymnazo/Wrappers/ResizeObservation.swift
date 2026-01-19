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
public struct ResizeObservation<BaseEnv: Env>: Env
    where BaseEnv.Observation == MLXArray
{
    public typealias Observation = MLXArray
    public typealias Action = BaseEnv.Action

    public var env: BaseEnv
    public let targetHeight: Int
    public let targetWidth: Int
    public let observationSpace: any Space<Observation>
    
    public var actionSpace: any Space<BaseEnv.Action> { env.actionSpace }
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
    public init(env: BaseEnv, shape: (Int, Int)) throws {
        self.env = env
        self.targetHeight = shape.0
        self.targetWidth = shape.1
        
        guard let innerBox = env.observationSpace as? Box,
              let innerShape = innerBox.shape,
              innerShape.count >= 2 else {
            throw GymnazoError.invalidResizeShape
        }
        
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
    
    public mutating func step(_ action: BaseEnv.Action) throws -> Step<MLXArray> {
        let result = try env.step(action)
        return Step(
            obs: resize(result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
    
    public mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset<MLXArray> {
        let result = try env.reset(seed: seed, options: options)
        return Reset(obs: resize(result.obs), info: result.info)
    }
    
    public var unwrapped: any Env { env.unwrapped }
    
    @discardableResult
    public func render() throws -> RenderOutput? { try env.render() }
    
    public mutating func close() { env.close() }
    
    private func resize(_ obs: MLXArray) -> MLXArray {
        let originalShape = obs.shape
        
        let inputNdim = originalShape.count
        
        var input = obs.asType(.float32)
        
        if inputNdim == 2 {
            input = input.expandedDimensions(axis: 0).expandedDimensions(axis: -1)
        } else if inputNdim == 3 {
            input = input.expandedDimensions(axis: 0)
        }
        
        let scaleH = Float(originalShape[0]) / Float(targetHeight)
        let scaleW = Float(originalShape[1]) / Float(targetWidth)
        
        var indices_h = [Int32]()
        var indices_w = [Int32]()
        
        for h in 0..<targetHeight {
            let srcH = min(Int(Float(h) * scaleH), originalShape[0] - 1)
            indices_h.append(Int32(srcH))
        }
        
        for w in 0..<targetWidth {
            let srcW = min(Int(Float(w) * scaleW), originalShape[1] - 1)
            indices_w.append(Int32(srcW))
        }
        
        let indicesHArray = MLXArray(indices_h)
        let indicesWArray = MLXArray(indices_w)
        
        var resized = input.take(indicesHArray, axis: 1)
        resized = resized.take(indicesWArray, axis: 2)
        
        resized = resized.squeezed(axis: 0)
        
        if inputNdim == 2 {
            resized = resized.squeezed(axis: -1)
        }
        
        return resized.asType(observationSpace.dtype ?? .uint8)
    }
}
