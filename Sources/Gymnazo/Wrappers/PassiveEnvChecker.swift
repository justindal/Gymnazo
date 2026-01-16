//
// PassiveEnvChecker.swift
//

/// Wrapper that runs passive validation on the first reset/step/render call to ensure
/// the environment conforms to Gymnazo's API expectations.
public final class PassiveEnvChecker<BaseEnv: Env>: Wrapper {
    public var env: BaseEnv

    private var cachedSpec: EnvSpec?
    private var checkedReset = false
    private var checkedStep = false
    private var checkedRender = false
    private var closeCalled = false

    public required init(env: BaseEnv) {
        self.env = env
        PassiveEnvChecks.ensureSpacesExist(for: env)
    }

    public func step(_ action: BaseEnv.Action) -> Step<BaseEnv.Observation> {
        if !checkedStep {
            checkedStep = true
            return PassiveEnvChecks.step(env: &env, action: action)
        }
        return env.step(action)
    }

    public func reset(seed: UInt64?, options: [String : Any]?) -> Reset<BaseEnv.Observation> {
        if !checkedReset {
            checkedReset = true
            cachedSpec = nil
            return PassiveEnvChecks.reset(env: &env, seed: seed, options: options)
        }
        return env.reset(seed: seed, options: options)
    }

    @discardableResult
    public func render() -> Any? {
        if !checkedRender {
            checkedRender = true
            PassiveEnvChecks.render(env: &env)
        }
        return env.render()
    }

    public func close() {
        if !closeCalled {
            closeCalled = true
            env.close()
            return
        }
        env.close()
    }

    public var spec: EnvSpec? {
        get {
            if let cachedSpec {
                return cachedSpec
            }

            guard var envSpec = env.spec else {
                return nil
            }

            envSpec.disable_env_checker = false
            cachedSpec = envSpec
            return envSpec
        }
        set {
            cachedSpec = nil
            env.spec = newValue
        }
    }
}
