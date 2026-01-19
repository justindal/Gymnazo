public struct AnyEnv<Observation, Action>: Env {
    private var box: AnyEnvBoxBase<Observation, Action>

    public init<E: Env>(_ env: E) where E.Observation == Observation, E.Action == Action {
        self.box = AnyEnvBox(env)
    }

    public var observationSpace: any Space<Observation> {
        box.observationSpace
    }

    public var actionSpace: any Space<Action> {
        box.actionSpace
    }

    public var spec: EnvSpec? {
        get { box.spec }
        set { box.spec = newValue }
    }

    public var renderMode: RenderMode? {
        get { box.renderMode }
        set { box.renderMode = newValue }
    }

    public var unwrapped: any Env {
        box.unwrapped
    }

    public mutating func step(_ action: Action) throws -> Step<Observation> {
        try box.step(action)
    }

    public mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset<Observation> {
        try box.reset(seed: seed, options: options)
    }

    @discardableResult
    public func render() throws -> RenderOutput? {
        try box.render()
    }

    public mutating func close() {
        box.close()
    }
}

private class AnyEnvBoxBase<Observation, Action> {
    var observationSpace: any Space<Observation> {
        fatalError("Not implemented")
    }

    var actionSpace: any Space<Action> {
        fatalError("Not implemented")
    }

    var spec: EnvSpec? {
        get { nil }
        set {}
    }

    var renderMode: RenderMode? {
        get { nil }
        set {}
    }

    var unwrapped: any Env {
        fatalError("Not implemented")
    }

    func step(_ action: Action) throws -> Step<Observation> {
        fatalError("Not implemented")
    }

    func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset<Observation> {
        fatalError("Not implemented")
    }

    func render() throws -> RenderOutput? {
        nil
    }

    func close() {}
}

private final class AnyEnvBox<E: Env>: AnyEnvBoxBase<E.Observation, E.Action> {
    var env: E

    init(_ env: E) {
        self.env = env
    }

    override var observationSpace: any Space<E.Observation> {
        env.observationSpace
    }

    override var actionSpace: any Space<E.Action> {
        env.actionSpace
    }

    override var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }

    override var renderMode: RenderMode? {
        get { env.renderMode }
        set { env.renderMode = newValue }
    }

    override var unwrapped: any Env {
        env.unwrapped
    }

    override func step(_ action: E.Action) throws -> Step<E.Observation> {
        try env.step(action)
    }

    override func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset<E.Observation> {
        try env.reset(seed: seed, options: options)
    }

    override func render() throws -> RenderOutput? {
        try env.render()
    }

    override func close() {
        env.close()
    }
}
