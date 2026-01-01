//
//  FrameStackObservation.swift
//  Gymnazo
//
//  Stacks the last N observations for temporal information.
//

import MLX

/// Padding type for frame stacking.
public enum FrameStackPadding {
    case reset
    case zero
}

/// Stacks the last N observations for temporal information.
///
/// Essential for image-based environments where the agent needs to
/// perceive motion and velocity from pixel observations.
///
/// ## Example
/// ```swift
/// var env = CarRacing()
/// var stackedEnv = FrameStackObservation(env: env, stackSize: 4)
/// // Observation shape: [96, 96, 3] -> [4, 96, 96, 3]
/// ```
public struct FrameStackObservation<InnerEnv: Env>: Env
    where InnerEnv.Observation == MLXArray
{
    public typealias Observation = MLXArray
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = Box
    public typealias ActionSpace = InnerEnv.ActionSpace
    
    public var env: InnerEnv
    public let stackSize: Int
    public let paddingType: FrameStackPadding
    public let observation_space: Box
    
    /// Circular buffer of stacked frames
    private var frameBuffer: [MLXArray]
    private var bufferIndex: Int = 0
    private var frameShape: [Int]
    
    public var action_space: ActionSpace { env.action_space }
    public var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }
    public var render_mode: String? {
        get { env.render_mode }
        set { env.render_mode = newValue }
    }
    
    /// Creates a frame stack observation wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap
    ///   - stackSize: Number of frames to stack (typically 4)
    ///   - paddingType: How to pad initial frames: `.reset` or `.zero`
    public init(env: InnerEnv, stackSize: Int, paddingType: FrameStackPadding = .reset) {
        precondition(stackSize >= 1, "stackSize must be at least 1")
        
        self.env = env
        self.stackSize = stackSize
        self.paddingType = paddingType
        
        guard let innerBox = env.observation_space as? Box,
              let innerShape = innerBox.shape else {
            fatalError("FrameStackObservation requires Box observation space with defined shape")
        }
        
        self.frameShape = innerShape
        
        let newShape = [stackSize] + innerShape
        self.observation_space = Box(
            low: 0,
            high: 255,
            shape: newShape,
            dtype: innerBox.dtype ?? .uint8
        )
        
        self.frameBuffer = []
    }
    
    public mutating func step(_ action: Action) -> Step<Observation> {
        let result = env.step(action)
        
        addFrame(result.obs)
        
        return Step(
            obs: getStackedObservation(),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
    
    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<Observation> {
        let result = env.reset(seed: seed, options: options)
        
        frameBuffer = []
        
        let initialFrame: MLXArray
        switch paddingType {
        case .reset:
            initialFrame = result.obs
        case .zero:
            initialFrame = MLXArray.zeros(frameShape).asType(observation_space.dtype ?? .uint8)
        }
        
        for _ in 0..<stackSize {
            frameBuffer.append(initialFrame)
        }
        
        if paddingType == .zero {
            frameBuffer[stackSize - 1] = result.obs
        }
        
        bufferIndex = 0
        
        return Reset(obs: getStackedObservation(), info: result.info)
    }
    
    public var unwrapped: any Env { env.unwrapped }
    
    @discardableResult
    public func render() -> Any? { env.render() }
    
    public func close() { env.close() }
    
    private mutating func addFrame(_ frame: MLXArray) {
        if frameBuffer.count < stackSize {
            frameBuffer.append(frame)
        } else {
            frameBuffer[bufferIndex] = frame
            bufferIndex = (bufferIndex + 1) % stackSize
        }
    }
    
    private func getStackedObservation() -> MLXArray {
        guard frameBuffer.count == stackSize else {
            let fullShape = [stackSize] + frameShape
            return MLXArray.zeros(fullShape).asType(observation_space.dtype ?? .uint8)
        }
        
        var orderedFrames: [MLXArray] = []
        for i in 0..<stackSize {
            let idx = (bufferIndex + i) % stackSize
            orderedFrames.append(frameBuffer[idx])
        }
        
        let stacked = MLX.stacked(orderedFrames, axis: 0)
        return stacked.asType(observation_space.dtype ?? .uint8)
    }
}
