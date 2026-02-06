//
// OrderEnforcing.swift
//

import MLX

/// Ensures `reset()` is called before `step()` or `render()`.
public final class OrderEnforcing: Wrapper {
    public var env: any Env

    private var hasReset = false
    private var cachedSpec: EnvSpec?
    private let disableRenderOrderEnforcing: Bool

    public required convenience init(env: any Env) {
        self.init(env: env, disableRenderOrderEnforcing: false)
    }

    public init(env: any Env, disableRenderOrderEnforcing: Bool) {
        self.env = env
        self.disableRenderOrderEnforcing = disableRenderOrderEnforcing
    }

    public func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        hasReset = true
        cachedSpec = nil
        return try env.reset(seed: seed, options: options)
    }

    public func step(_ action: MLXArray) throws -> Step {
        guard hasReset else {
            throw GymnazoError.stepBeforeReset
        }
        return try env.step(action)
    }

    @discardableResult
    public func render() throws -> RenderOutput? {
        guard disableRenderOrderEnforcing || hasReset else {
            throw GymnazoError.renderBeforeReset
        }
        return try env.render()
    }

    public var spec: EnvSpec? {
        get {
            if let cachedSpec {
                return cachedSpec
            }

            guard var envSpec = env.spec else {
                return nil
            }

            envSpec.orderEnforce = true
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

