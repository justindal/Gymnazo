//
//  ActionBoxWrappers.swift
//

import MLX

/// Clips actions to the Box action space bounds before passing to the inner environment.
public final class ClipAction: ActionWrapper {
    public override func action(_ action: MLXArray) -> MLXArray {
        guard let space = env.actionSpace as? Box else {
            fatalError("ClipAction requires a Box action space")
        }
        let clippedLow = MLX.maximum(action, space.low)
        let clipped = MLX.minimum(clippedLow, space.high)
        return clipped
    }
}

/// Rescales actions from a source range (default [-1, 1]) into the Box action space bounds.
public final class RescaleAction: ActionWrapper {
    private let srcLow: Float
    private let srcHigh: Float

    public init(env: any Env, sourceLow: Float = -1.0, sourceHigh: Float = 1.0) {
        self.srcLow = sourceLow
        self.srcHigh = sourceHigh
        super.init(env: env)
    }

    public required init(env: any Env) {
        self.srcLow = -1.0
        self.srcHigh = 1.0
        super.init(env: env)
    }

    public override func action(_ action: MLXArray) -> MLXArray {
        guard let space = env.actionSpace as? Box else {
            fatalError("RescaleAction requires a Box action space")
        }
        let srcLowArr = MLXArray(srcLow).asType(space.dtype ?? .float32)
        let srcHighArr = MLXArray(srcHigh).asType(space.dtype ?? .float32)
        let spanSrc = srcHighArr - srcLowArr
        let spanTgt = space.high - space.low
        let scaled = space.low + (action - srcLowArr) * spanTgt / spanSrc
        let clippedLow = MLX.maximum(scaled, space.low)
        let clipped = MLX.minimum(clippedLow, space.high)
        return clipped
    }
}


