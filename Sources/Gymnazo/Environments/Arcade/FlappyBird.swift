import MLX

/// A side-scrolling control environment inspired by Flappy Bird.
///
/// The agent observes normalized bird height, vertical velocity, horizontal distance to the next
/// pipe, and the next gap's vertical center. Action zero coasts and action one flaps. Each safe
/// step rewards `0.1`, passing a pipe adds `1`, and a collision rewards `-1` and ends the episode.
public struct FlappyBird: Env {
    public let actionSpace: any Space
    public let observationSpace: any Space

    public var spec: EnvSpec?
    public var renderMode: RenderMode?

    private var model = FlappyBirdModel()

    /// Creates the standard Flappy Bird environment.
    ///
    /// - Parameter renderMode: Optional rendering mode.
    public init(renderMode: RenderMode? = nil) {
        self.renderMode = renderMode
        self.actionSpace = Discrete(n: 2)
        self.observationSpace = Box(low: -1, high: 1, shape: [4])
    }

    public mutating func step(_ action: MLXArray) throws -> Step {
        let value = Int(action.singletonValue(Int32.self))
        guard value == 0 || value == 1 else {
            throw GymnazoError.invalidAction("FlappyBird actions must be 0 or 1.")
        }

        let result = try model.step(flaps: value == 1)
        return Step(
            obs: MLXArray(result.observation),
            reward: result.reward,
            terminated: result.terminated,
            truncated: false,
            info: ["score": InfoValue(result.score)]
        )
    }

    public mutating func reset(
        seed: UInt64? = nil,
        options: EnvOptions? = nil
    ) throws -> Reset {
        guard options?.isEmpty != false else {
            throw GymnazoError.invalidConfiguration(
                "FlappyBird does not support reset options."
            )
        }
        return Reset(obs: MLXArray(model.reset(seed: seed)), info: ["score": 0])
    }

    public func render() throws -> RenderOutput? {
        guard let renderMode else { return nil }
        switch renderMode {
        case .human:
            return .other(currentSnapshot)
        case .rgbArray:
            #if canImport(CoreGraphics)
                guard let image = FlappyBirdImageRenderer.image(snapshot: currentSnapshot) else {
                    throw GymnazoError.invalidState("FlappyBird could not create a render frame.")
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
    public var currentSnapshot: FlappyBirdSnapshot {
        model.snapshot
    }
}

/// A render-safe snapshot of a Flappy Bird episode.
public struct FlappyBirdSnapshot: Sendable, Equatable {
    public let birdY: Float
    public let birdVelocity: Float
    public let pipes: [FlappyBirdPipeSnapshot]
    public let score: Int

    public static let zero = FlappyBirdSnapshot(
        birdY: 0.5,
        birdVelocity: 0,
        pipes: [],
        score: 0
    )
}

/// A render-safe snapshot of one pipe and its gap.
public struct FlappyBirdPipeSnapshot: Sendable, Equatable {
    public let x: Float
    public let gapY: Float
}
