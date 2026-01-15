/// A wrapper protocol for environments that change their observation and/or action types.
public protocol TransformingWrapper<BaseEnv>: Env {
    associatedtype BaseEnv: Env
    var env: BaseEnv { get set }
    init(env: BaseEnv)
}

public extension TransformingWrapper {
    var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }

    var renderMode: String? {
        get { env.renderMode }
        set { env.renderMode = newValue }
    }

    var unwrapped: any Env {
        env.unwrapped
    }

    func close() {
        env.close()
    }

    @discardableResult
    func render() -> Any? {
        env.render()
    }
}

public extension TransformingWrapper where ActionSpace == BaseEnv.ActionSpace {
    var actionSpace: BaseEnv.ActionSpace { env.actionSpace }
}

