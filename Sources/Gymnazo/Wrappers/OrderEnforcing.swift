//
// OrderEnforcing.swift
//

/// Ensures `reset()` is called before `step()` or `render()`.
public final class OrderEnforcing<BaseEnv: Env>: Wrapper {
    public var env: BaseEnv

    private var hasReset = false
    private var cachedSpec: EnvSpec?
    private let disableRenderOrderEnforcing: Bool

    public required convenience init(env: BaseEnv) {
        self.init(env: env, disableRenderOrderEnforcing: false)
    }

    public init(env: BaseEnv, disableRenderOrderEnforcing: Bool) {
        self.env = env
        self.disableRenderOrderEnforcing = disableRenderOrderEnforcing
    }

    public func reset(seed: UInt64?, options: [String : Any]?) -> Reset<BaseEnv.Observation> {
        hasReset = true
        cachedSpec = nil
        return env.reset(seed: seed, options: options)
    }

    public func step(_ action: BaseEnv.Action) -> Step<BaseEnv.Observation> {
        guard hasReset else {
            fatalError("OrderEnforcing: Cannot call env.step() before env.reset().")
        }
        return env.step(action)
    }

    @discardableResult
    public func render() -> Any? {
        guard disableRenderOrderEnforcing || hasReset else {
            fatalError("OrderEnforcing: Cannot call env.render() before env.reset().")
        }
        return env.render()
    }

    public var spec: EnvSpec? {
        get {
            if let cachedSpec {
                return cachedSpec
            }

            guard var envSpec = env.spec else {
                return nil
            }

            envSpec.order_enforce = true
            cachedSpec = envSpec
            return envSpec
        }
        set {
            cachedSpec = nil
            env.spec = newValue
        }
    }

    public var hasResetFlag: Bool {
        hasReset
    }
}
