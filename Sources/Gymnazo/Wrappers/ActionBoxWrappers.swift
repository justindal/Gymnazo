//
//  ActionBoxWrappers.swift
//

import MLX

/// clips actions to the Box action space bounds before passing to the inner environment
public final class ClipAction<BaseEnv: Env>: ActionWrapper<BaseEnv> where BaseEnv.ActionSpace == Box {
    public override func action(_ action: Action) -> Action {
        let space = env.actionSpace
        let clippedLow = MLX.maximum(action, space.low)
        let clipped = MLX.minimum(clippedLow, space.high)
        return clipped
    }
}

/// rescales actions from a source range (default [-1, 1]) into the Box action space bounds.
//  after rescaling, actions are clipped to the Box bounds
public final class RescaleAction<BaseEnv: Env>: ActionWrapper<BaseEnv> where BaseEnv.ActionSpace == Box {
    private let srcLow: Float
    private let srcHigh: Float

    public init(env: BaseEnv, sourceLow: Float = -1.0, sourceHigh: Float = 1.0) {
        self.srcLow = sourceLow
        self.srcHigh = sourceHigh
        super.init(env: env)
    }

    public required init(env: BaseEnv) {
        self.srcLow = -1.0
        self.srcHigh = 1.0
        super.init(env: env)
    }

    public override func action(_ action: Action) -> Action {
        let space = env.actionSpace
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

