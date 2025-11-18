//
// OrderEnforcing.swift
//

/// Ensures `reset()` is called before `step()` or `render()`.
public final class OrderEnforcing<InnerEnv: Environment>: Wrapper {
    public typealias Observation = InnerEnv.Observation
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = InnerEnv.ObservationSpace
    public typealias ActionSpace = InnerEnv.ActionSpace

    public var env: InnerEnv

    private var hasReset = false
    private var cachedSpec: EnvSpec?
    private let disableRenderOrderEnforcing: Bool

    public required convenience init(env: InnerEnv) {
        self.init(env: env, disableRenderOrderEnforcing: false)
    }

    public init(env: InnerEnv, disableRenderOrderEnforcing: Bool) {
        self.env = env
        self.disableRenderOrderEnforcing = disableRenderOrderEnforcing
    }

    public func reset(seed: UInt64?, options: [String : Any]?) -> ResetResult {
        hasReset = true
        cachedSpec = nil
        return env.reset(seed: seed, options: options)
    }

    public func step(_ action: Action) -> StepResult {
        guard hasReset else {
            fatalError("OrderEnforcing: Cannot call env.step() before env.reset().")
        }
        return env.step(action)
    }

    public func render() {
        guard disableRenderOrderEnforcing || hasReset else {
            fatalError("OrderEnforcing: Cannot call env.render() before env.reset().")
        }
        env.render()
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
