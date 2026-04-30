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
    /// Fill initial frames with a custom observation, placing the reset observation last.
    case custom(MLXArray)
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
/// // Standard CarRacing preprocessing pipeline (channel-stacked for NatureCNN)
/// var env = CarRacing()
///     .grayscale()
///     .resized(to: (84, 84))
///     .frameStacked(4, stackAxis: -1)  // [84, 84] -> [84, 84, 4]
///     .timeLimited(1000)
/// ```
///
/// ## Topics
///
/// ### Creating a Frame Stack Wrapper
/// - ``init(env:stackSize:paddingType:stackAxis:)``
///
/// ### Configuration
/// - ``stackSize``
/// - ``paddingType``
/// - ``stackAxis``
public struct FrameStackObservation: Wrapper {
    public var env: any Env
    public let stackSize: Int
    public let stackAxis: Int
    public let paddingType: FrameStackPadding
    public let observationSpace: any Space

    private var frameBuffer: [MLXArray]
    private var frameShape: [Int]
    private var frameDtype: DType

    public var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }
    public var renderMode: RenderMode? {
        get { env.renderMode }
        set { env.renderMode = newValue }
    }

    /// Creates a frame stack observation wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap
    ///   - stackSize: Number of frames to stack (typically 4)
    ///   - paddingType: How to pad initial frames: `.reset` or `.zero`
    ///   - stackAxis: Axis along which to stack frames. Use `-1` for channel-last (HWC) layout.
    public init(
        env: any Env,
        stackSize: Int,
        paddingType: FrameStackPadding = .reset,
        stackAxis: Int = 0
    ) throws {
        guard stackSize >= 1 else {
            throw GymnazoError.invalidStackSize(stackSize)
        }

        self.env = env
        self.stackSize = stackSize
        self.stackAxis = stackAxis
        self.paddingType = paddingType

        guard let innerBox = env.observationSpace as? Box,
            let innerShape = innerBox.shape
        else {
            throw GymnazoError.invalidObservationSpace
        }

        self.frameShape = innerShape
        self.frameDtype = innerBox.dtype ?? .uint8

        if case .custom(let custom) = paddingType, custom.shape != innerShape {
            throw GymnazoError.invalidObservationSpace
        }

        let low = MLX.stacked(Array(repeating: innerBox.low, count: stackSize), axis: stackAxis)
        let high = MLX.stacked(Array(repeating: innerBox.high, count: stackSize), axis: stackAxis)
        self.observationSpace = Box(low: low, high: high, dtype: self.frameDtype)

        self.frameBuffer = []
    }

    public mutating func step(_ action: MLXArray) throws -> Step {
        let result = try env.step(action)

        addFrame(result.obs)

        return Step(
            obs: getStackedObservation(),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        let result = try env.reset(seed: seed, options: options)

        frameBuffer.removeAll(keepingCapacity: true)
        let paddingValue = paddingValue(from: result.obs)
        for _ in 0..<(stackSize - 1) {
            frameBuffer.append(paddingValue)
        }
        frameBuffer.append(result.obs)

        return Reset(obs: getStackedObservation(), info: result.info)
    }

    private mutating func addFrame(_ frame: MLXArray) {
        if frameBuffer.count == stackSize {
            frameBuffer.removeFirst()
        }
        frameBuffer.append(frame)
    }

    private func paddingValue(from resetObservation: MLXArray) -> MLXArray {
        switch paddingType {
        case .reset:
            return resetObservation
        case .zero:
            return MLXArray.zeros(frameShape, dtype: frameDtype)
        case .custom(let custom):
            return custom.asType(frameDtype)
        }
    }

    private func getStackedObservation() -> MLXArray {
        if frameBuffer.count == stackSize {
            return MLX.stacked(frameBuffer, axis: stackAxis).asType(frameDtype)
        }

        var padded = frameBuffer
        if padded.count < stackSize {
            let zero = MLXArray.zeros(frameShape, dtype: frameDtype)
            for _ in 0..<(stackSize - padded.count) {
                padded.insert(zero, at: 0)
            }
        }
        return MLX.stacked(padded, axis: stackAxis).asType(frameDtype)
    }
}
