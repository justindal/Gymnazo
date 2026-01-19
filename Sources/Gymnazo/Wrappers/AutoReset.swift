/// Automatically resets an environment when an episode ends.
///
/// The wrapper writes terminal observation and info into the info dictionary.
public struct AutoReset<BaseEnv: Env>: Wrapper {
    public var env: BaseEnv
    public let mode: AutoresetMode

    private var needsReset: Bool = true

    /// Creates the wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap.
    ///   - mode: Autoreset behavior (`nextStep`, `sameStep`, or `disabled`).
    public init(env: BaseEnv, mode: AutoresetMode = .nextStep) {
        self.env = env
        self.mode = mode
    }

    public init(env: BaseEnv) {
        self.init(env: env, mode: .nextStep)
    }

    public mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset<BaseEnv.Observation> {
        needsReset = false
        return try env.reset(seed: seed, options: options)
    }

    public mutating func step(_ action: BaseEnv.Action) throws -> Step<BaseEnv.Observation> {
        if needsReset && mode == .nextStep {
            _ = try env.reset(seed: nil, options: nil)
            needsReset = false
        }

        let result = try env.step(action)
        let done = result.terminated || result.truncated

        if !done {
            return result
        }

        if mode == .disabled {
            return result
        }

        if mode == .nextStep {
            needsReset = true
            var info = result.info
            if let value = sendableValue(result.obs) {
                info["final_observation"] = value
            }
            info["final_info"] = .object(result.info.storage)
            return Step(
                obs: result.obs,
                reward: result.reward,
                terminated: result.terminated,
                truncated: result.truncated,
                info: info
            )
        }

        let terminalObs = result.obs
        let terminalInfo = result.info
        let resetResult = try env.reset(seed: nil, options: nil)
        needsReset = false

        var info = resetResult.info
        if let value = sendableValue(terminalObs) {
            info["final_observation"] = value
        }
        info["final_info"] = .object(terminalInfo.storage)

        return Step(
            obs: resetResult.obs,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: info
        )
    }
}

private func sendableValue<Observation>(_ value: Observation) -> InfoValue? {
    switch value {
    case let v as Bool:
        return .bool(v)
    case let v as Int:
        return .int(v)
    case let v as Float:
        return .double(Double(v))
    case let v as Double:
        return .double(v)
    case let v as String:
        return .string(v)
    case let v as [InfoValue]:
        return .array(v)
    case let v as [String: InfoValue]:
        return .object(v)
    default:
        return nil
    }
}
