import MLX

/// A grid-based Snake environment with deterministic food placement.
///
/// The observation contains the complete normalized 12-by-12 board followed by a four-value
/// one-hot heading. Action zero continues straight, action one turns right, and action two turns
/// left. Eating food rewards `1`, each movement step rewards `-0.01`, and collisions reward `-1`.
public struct Snake: Env {
    public let actionSpace: any Space
    public let observationSpace: any Space

    public var spec: EnvSpec?
    public var renderMode: RenderMode?

    private var model = SnakeModel()

    /// Creates the standard 12-by-12 Snake environment.
    ///
    /// - Parameter renderMode: Optional rendering mode.
    public init(renderMode: RenderMode? = nil) {
        self.renderMode = renderMode
        self.actionSpace = Discrete(n: 3)
        self.observationSpace = Box(
            low: 0,
            high: 1,
            shape: [SnakeModel.observationCount]
        )
    }

    public mutating func step(_ action: MLXArray) throws -> Step {
        let value = Int(action.singletonValue(Int32.self))
        guard (0..<3).contains(value) else {
            throw GymnazoError.invalidAction("Snake actions must be 0, 1, or 2.")
        }

        let result = try model.step(action: value)
        return Step(
            obs: MLXArray(result.observation),
            reward: result.reward,
            terminated: result.terminated,
            truncated: false,
            info: [
                "score": InfoValue(result.score),
                "won": InfoValue(result.won),
            ]
        )
    }

    public mutating func reset(
        seed: UInt64? = nil,
        options: EnvOptions? = nil
    ) throws -> Reset {
        guard options?.isEmpty != false else {
            throw GymnazoError.invalidConfiguration("Snake does not support reset options.")
        }
        return Reset(
            obs: MLXArray(model.reset(seed: seed)),
            info: ["score": 0, "won": false]
        )
    }

    public func render() throws -> RenderOutput? {
        guard let renderMode else { return nil }
        switch renderMode {
        case .human:
            return .other(currentSnapshot)
        case .rgbArray:
            #if canImport(CoreGraphics)
                guard let image = SnakeImageRenderer.image(snapshot: currentSnapshot) else {
                    throw GymnazoError.invalidState("Snake could not create a render frame.")
                }
                return .rgbArray(image)
            #else
                return nil
            #endif
        case .ansi, .statePixels:
            return nil
        }
    }

    /// The immutable state used by renderers.
    public var currentSnapshot: SnakeSnapshot {
        model.snapshot
    }
}

/// A coordinate in a Snake board.
public struct SnakeCell: Sendable, Hashable {
    public let x: Int
    public let y: Int

    /// Creates a board coordinate.
    public init(x: Int, y: Int) {
        self.x = x
        self.y = y
    }
}

/// A render-safe snapshot of a Snake episode.
public struct SnakeSnapshot: Sendable, Equatable {
    public let columns: Int
    public let rows: Int
    public let body: [SnakeCell]
    public let food: SnakeCell?
    public let score: Int

    public static let zero = SnakeSnapshot(
        columns: 12,
        rows: 12,
        body: [],
        food: nil,
        score: 0
    )
}
