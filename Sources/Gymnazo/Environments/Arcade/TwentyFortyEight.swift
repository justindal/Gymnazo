import MLX

public struct TwentyFortyEight: Env {
    public let actionSpace: any Space
    public let observationSpace: any Space

    public var spec: EnvSpec?
    public var renderMode: RenderMode?

    private var model = TwentyFortyEightModel()

    public init(renderMode: RenderMode? = nil) {
        self.renderMode = renderMode
        self.actionSpace = Discrete(n: 4)
        self.observationSpace = Box(
            low: 0,
            high: 1,
            shape: [TwentyFortyEightModel.observationCount]
        )
    }

    public mutating func step(_ action: MLXArray) throws -> Step {
        guard actionSpace.contains(action) else {
            throw GymnazoError.invalidAction(
                "2048 expects a single integer action in 0...3."
            )
        }
        let value = Int(action.singletonValue(Int32.self))
        let result = try model.step(action: value)
        return Step(
            obs: MLXArray(result.observation),
            reward: result.reward,
            terminated: result.terminated,
            truncated: false,
            info: [
                "score": InfoValue(result.score),
                "max_tile": InfoValue(result.maxTile),
                "game_over": InfoValue(result.terminated),
            ]
        )
    }

    public mutating func reset(
        seed: UInt64? = nil,
        options: EnvOptions? = nil
    ) throws -> Reset {
        guard options?.isEmpty != false else {
            throw GymnazoError.invalidConfiguration(
                "2048 does not support reset options."
            )
        }
        let observation = model.reset(seed: seed)
        let snapshot = model.snapshot
        return Reset(
            obs: MLXArray(observation),
            info: [
                "score": InfoValue(snapshot.score),
                "max_tile": InfoValue(snapshot.maxTile),
                "game_over": InfoValue(snapshot.isGameOver),
            ]
        )
    }

    public func render() throws -> RenderOutput? {
        guard let renderMode else { return nil }
        switch renderMode {
        case .human:
            return .other(currentSnapshot)
        case .rgbArray:
            #if canImport(CoreGraphics)
                guard let image = TwentyFortyEightImageRenderer.image(snapshot: currentSnapshot)
                else {
                    throw GymnazoError.invalidState("2048 could not create a render frame.")
                }
                return .rgbArray(image)
            #else
                return nil
            #endif
        case .ansi, .statePixels:
            return nil
        }
    }

    public var currentSnapshot: TwentyFortyEightSnapshot {
        model.snapshot
    }
}

public struct TwentyFortyEightSnapshot: Sendable, Equatable {
    public let tiles: [Int]
    public let score: Int
    public let maxTile: Int
    public let isGameOver: Bool

    public static let zero = TwentyFortyEightSnapshot(
        tiles: [Int](repeating: 0, count: TwentyFortyEightModel.cellCount),
        score: 0,
        maxTile: 0,
        isGameOver: false
    )

    public init(tiles: [Int], score: Int, maxTile: Int, isGameOver: Bool) {
        self.tiles = tiles
        self.score = score
        self.maxTile = maxTile
        self.isGameOver = isGameOver
    }
}
