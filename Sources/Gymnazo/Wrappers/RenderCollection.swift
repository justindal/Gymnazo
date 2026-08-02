import MLX

public struct RenderCollection: Wrapper {
    public var env: any Env
    public var renderMode: RenderMode?
    private var frames: [RenderOutput] = []

    public init(env: any Env) {
        self.env = env
        self.renderMode = env.renderMode
    }

    public mutating func reset(
        seed: UInt64?,
        options: EnvOptions?
    ) throws -> Reset {
        frames.removeAll(keepingCapacity: true)
        let result = try env.reset(seed: seed, options: options)
        try capture()
        return result
    }

    public mutating func step(_ action: MLXArray) throws -> Step {
        let result = try env.step(action)
        try capture()
        return result
    }

    public mutating func render() throws -> RenderOutput? {
        let result = RenderOutput.collection(frames)
        frames.removeAll(keepingCapacity: true)
        return result
    }

    private mutating func capture() throws {
        if let frame = try env.render() {
            frames.append(frame)
        }
    }
}
