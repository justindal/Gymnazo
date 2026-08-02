import MLX

/// Collects render outputs until the caller requests a frame collection.
public struct RenderCollection: Wrapper {
    public var env: any Env
    public var renderMode: RenderMode?
    private var frames: [RenderOutput] = []

    /// Creates a render-collecting wrapper around an environment.
    ///
    /// - Parameter env: Environment whose frames should be collected.
    public init(env: any Env) {
        self.env = env
        self.renderMode = env.renderMode
    }

    /// Resets the wrapped environment and captures its initial frame.
    ///
    /// - Parameters:
    ///   - seed: Optional deterministic reset seed.
    ///   - options: Environment-specific reset options.
    /// - Returns: The wrapped reset result.
    /// - Throws: If reset or frame capture fails.
    public mutating func reset(
        seed: UInt64?,
        options: EnvOptions?
    ) throws -> Reset {
        frames.removeAll(keepingCapacity: true)
        let result = try env.reset(seed: seed, options: options)
        try capture()
        return result
    }

    /// Steps the wrapped environment and captures its frame.
    ///
    /// - Parameter action: MLX action passed to the wrapped environment.
    /// - Returns: The wrapped step result.
    /// - Throws: If stepping or frame capture fails.
    public mutating func step(_ action: MLXArray) throws -> Step {
        let result = try env.step(action)
        try capture()
        return result
    }

    /// Returns all captured frames and clears the collection.
    ///
    /// - Returns: A collection output, including an empty collection when no frame was captured.
    /// - Throws: If the wrapped environment cannot render.
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
