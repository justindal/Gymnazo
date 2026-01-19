/// A wrapper protocol for environments that change their observation and/or action types.
public protocol TransformingWrapper<BaseEnv>: Env {
    associatedtype BaseEnv: Env
    var env: BaseEnv { get set }
}

public extension TransformingWrapper {
    var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }

    var renderMode: RenderMode? {
        get { env.renderMode }
        set { env.renderMode = newValue }
    }

    var unwrapped: any Env {
        env.unwrapped
    }

    mutating func close() {
        env.close()
    }

    @discardableResult
    func render() throws -> RenderOutput? {
        try env.render()
    }
}
