import MLX

@MainActor
public final class AsyncVectorEnv: VectorEnv {
    private let storage: SyncVectorEnv

    public var numEnvs: Int { storage.numEnvs }
    public var singleObservationSpace: any Space { storage.singleObservationSpace }
    public var singleActionSpace: any Space { storage.singleActionSpace }
    public var observationSpace: any Space { storage.observationSpace }
    public var actionSpace: any Space { storage.actionSpace }
    public var spec: EnvSpec? {
        get { storage.spec }
        set { storage.spec = newValue }
    }
    public var renderMode: RenderMode? { storage.renderMode }
    public var autoresetMode: AutoresetMode { storage.autoresetMode }
    public var closed: Bool { storage.closed }

    public init(
        envFns: [() -> any Env],
        copyObservations: Bool = true,
        autoresetMode: AutoresetMode = .nextStep
    ) throws {
        storage = try SyncVectorEnv(
            envFns: envFns,
            copyObservations: copyObservations,
            autoresetMode: autoresetMode
        )
    }

    public convenience init(
        envs: [any Env],
        copyObservations: Bool = true,
        autoresetMode: AutoresetMode = .nextStep
    ) throws {
        try self.init(
            envFns: envs.map { environment in { environment } },
            copyObservations: copyObservations,
            autoresetMode: autoresetMode
        )
    }

    public func step(_ actions: [MLXArray]) throws -> VectorStepResult {
        try storage.step(actions)
    }

    public func reset(seed: UInt64? = nil, options: EnvOptions? = nil) throws -> VectorResetResult {
        try storage.reset(seed: seed, options: options)
    }

    public func stepAsync(_ actions: [MLXArray]) async throws -> VectorStepResult {
        try step(actions)
    }

    public func resetAsync(seed: UInt64? = nil, options: EnvOptions? = nil)
        async throws -> VectorResetResult
    {
        try reset(seed: seed, options: options)
    }

    public func close() {
        storage.close()
    }
}
