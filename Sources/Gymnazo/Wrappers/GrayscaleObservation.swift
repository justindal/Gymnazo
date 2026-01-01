//
//  GrayscaleObservation.swift
//  Gymnazo
//
//  Converts RGB image observations to grayscale.
//

import MLX

/// Converts RGB image observations to grayscale.
///
/// Uses standard luminance weights (ITU-R BT.601):
/// `grayscale = 0.299*R + 0.587*G + 0.114*B`
///
/// ## Example
/// ```swift
/// var env = CarRacing()
/// var grayEnv = GrayscaleObservation(env: env)
/// // Observation shape: [96, 96, 3] -> [96, 96] or [96, 96, 1]
/// ```
public struct GrayscaleObservation<InnerEnv: Env>: Env
    where InnerEnv.Observation == MLXArray
{
    public typealias Observation = MLXArray
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = Box
    public typealias ActionSpace = InnerEnv.ActionSpace
    
    public var env: InnerEnv
    public let keepDim: Bool
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
    
    /// Creates a grayscale observation wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap (must have MLXArray observations with shape [H, W, 3])
    ///   - keepDim: If true, output shape is [H, W, 1]. If false, output shape is [H, W].
    public init(env: InnerEnv, keepDim: Bool = false) {
        self.env = env
        self.keepDim = keepDim
        
        guard let innerBox = env.observation_space as? Box,
              let shape = innerBox.shape,
              shape.count == 3,
              shape[2] == 3 else {
            fatalError("GrayscaleObservation requires Box observation space with shape [H, W, 3]")
        }
        
        let newShape: [Int] = keepDim ? [shape[0], shape[1], 1] : [shape[0], shape[1]]
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
            obs: toGrayscale(result.obs),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
    
    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<Observation> {
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
        
        return result.asType(observation_space.dtype ?? .uint8)
    }
}
