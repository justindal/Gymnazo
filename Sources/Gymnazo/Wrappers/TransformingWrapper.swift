public protocol TransformingWrapper: Env {
    associatedtype InnerEnv: Env
    var env: InnerEnv { get set }
    init(env: InnerEnv)
}

public extension TransformingWrapper {
    var spec: EnvSpec? {
        get { env.spec }
        set { env.spec = newValue }
    }

    var render_mode: String? {
        get { env.render_mode }
        set { env.render_mode = newValue }
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

public extension TransformingWrapper where ActionSpace == InnerEnv.ActionSpace {
    var action_space: InnerEnv.ActionSpace {
        env.action_space
    }
}

