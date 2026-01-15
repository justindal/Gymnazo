//
//  FrameStackObservation.swift
//  Gymnazo
//
//  Stacks the last N observations for temporal information.
//

import MLX

/// Padding strategy for frame stacking.
///
/// Determines how initial frames are filled before enough observations
/// have been collected to fill the stack.
public enum FrameStackPadding {
    /// Repeat the reset observation for all initial frames.
    case reset
    /// Fill initial frames with zeros, placing the reset observation last.
    case zero
}

/// Stacks the last N observations for temporal information.
///
/// This wrapper is essential for image-based environments where the agent needs to
/// perceive motion and velocity from static pixel observations.
///
/// ## Overview
///
/// `FrameStackObservation` maintains a circular buffer of the most recent observations.
/// By providing multiple consecutive frames to the agent, it enables:
/// - Detection of motion direction and speed
/// - Understanding of temporal dynamics
/// - Better state estimation in partially observable environments
///
/// ## Example
///
/// ```swift
/// // Standard CarRacing preprocessing pipeline
/// var env = CarRacing()
///     .grayscale()
///     .resized(to: (84, 84))
///     .frameStacked(4)           // [84, 84] -> [4, 84, 84]
///     .timeLimited(1000)
/// ```
///
/// ## Topics
///
/// ### Creating a Frame Stack Wrapper
/// - ``init(env:stackSize:paddingType:)``
///
/// ### Configuration
/// - ``stackSize``
/// - ``paddingType``
public struct FrameStackObservation<BaseEnv: Env>: Env
where BaseEnv.Observation == MLXArray {
    public var env: BaseEnv
    public let stackSize: Int
    public let paddingType: FrameStackPadding
    public let observationSpace: Box

    /// Circular buffer of stacked frames
    private var frameBuffer: [MLXArray]
    private var bufferIndex: Int = 0
    private var frameShape: [Int]

    public var actionSpace: BaseEnv.ActionSpace { env.actionSpace }
    public var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }
    public var renderMode: String? {
        get { env.renderMode }
        set { env.renderMode = newValue }
    }

    /// Creates a frame stack observation wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap
    ///   - stackSize: Number of frames to stack (typically 4)
    ///   - paddingType: How to pad initial frames: `.reset` or `.zero`
    public init(env: BaseEnv, stackSize: Int, paddingType: FrameStackPadding = .reset) {
        precondition(stackSize >= 1, "stackSize must be at least 1")

        self.env = env
        self.stackSize = stackSize
        self.paddingType = paddingType

        guard let innerBox = env.observationSpace as? Box,
            let innerShape = innerBox.shape
        else {
            fatalError("FrameStackObservation requires Box observation space with defined shape")
        }

        self.frameShape = innerShape

        let newShape = [stackSize] + innerShape
        self.observationSpace = Box(
            low: 0,
            high: 255,
            shape: newShape,
            dtype: innerBox.dtype ?? .uint8
        )

        self.frameBuffer = []
    }

    public mutating func step(_ action: BaseEnv.Action) -> Step<MLXArray> {
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

    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<MLXArray> {
        let result = env.reset(seed: seed, options: options)

        frameBuffer = []

        let initialFrame: MLXArray
        switch paddingType {
        case .reset:
            initialFrame = result.obs
        case .zero:
            initialFrame = MLXArray.zeros(frameShape).asType(observationSpace.dtype ?? .uint8)
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
            return MLXArray.zeros(fullShape).asType(observationSpace.dtype ?? .uint8)
        }

        var orderedFrames: [MLXArray] = []
        for i in 0..<stackSize {
            let idx = (bufferIndex + i) % stackSize
            orderedFrames.append(frameBuffer[idx])
        }

        let stacked = MLX.stacked(orderedFrames, axis: 0)
        return stacked.asType(observationSpace.dtype ?? .uint8)
    }
}
