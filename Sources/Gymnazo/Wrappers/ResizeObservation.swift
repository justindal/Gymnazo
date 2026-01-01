//
//  ResizeObservation.swift
//  Gymnazo
//
//  Resizes image observations to a specified shape.
//

import MLX

/// Resizes image observations to a specified shape.
///
/// Uses bilinear interpolation for resizing.
///
/// ## Example
/// ```swift
/// var env = CarRacing()
/// var resizedEnv = ResizeObservation(env: env, shape: (84, 84))
/// // Observation shape: [96, 96, 3] -> [84, 84, 3]
/// ```
public struct ResizeObservation<InnerEnv: Env>: Env
    where InnerEnv.Observation == MLXArray
{
    public typealias Observation = MLXArray
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = Box
    public typealias ActionSpace = InnerEnv.ActionSpace
    
    public var env: InnerEnv
    public let targetHeight: Int
    public let targetWidth: Int
    public let observation_space: Box
    
    public var action_space: ActionSpace { env.action_space }
    public var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }
    public var render_mode: String? {
        get { env.render_mode }
        set { env.render_mode = newValue }
    }
    
    /// Creates a resize observation wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap (must have MLXArray observations with shape [H, W, ...])
    ///   - shape: Target (height, width) for the resized observations
    public init(env: InnerEnv, shape: (Int, Int)) {
        self.env = env
        self.targetHeight = shape.0
        self.targetWidth = shape.1
        
        // Compute new observation space
        guard let innerBox = env.observation_space as? Box,
              let innerShape = innerBox.shape,
              innerShape.count >= 2 else {
            fatalError("ResizeObservation requires Box observation space with at least 2 dimensions")
        }
        
        var newShape = [targetHeight, targetWidth]
        if innerShape.count > 2 {
            newShape.append(contentsOf: innerShape[2...])
        }
        
        self.observation_space = Box(
            low: 0,
            high: 255,
            shape: newShape,
            dtype: innerBox.dtype ?? .uint8
        )
    }
    
    public mutating func step(_ action: Action) -> Step<Observation> {
        let result = env.step(action)
        return Step(
            obs: resize(result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
    
    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<Observation> {
        let result = env.reset(seed: seed, options: options)
        return Reset(obs: resize(result.obs), info: result.info)
    }
    
    public var unwrapped: any Env { env.unwrapped }
    
    @discardableResult
    public func render() -> Any? { env.render() }
    
    public func close() { env.close() }
    
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
        
        return resized.asType(observation_space.dtype ?? .uint8)
    }
}
