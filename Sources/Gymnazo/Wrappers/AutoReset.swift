import MLX

/// Automatically resets an environment when an episode ends.
///
/// The wrapper writes terminal observation and info into the info dictionary.
public struct AutoReset: Wrapper {
    public var env: any Env
    public let mode: AutoresetMode

    private var needsReset: Bool = true

    /// Creates the wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap.
    ///   - mode: Autoreset behavior (`nextStep`, `sameStep`, or `disabled`).
    public init(env: any Env, mode: AutoresetMode = .nextStep) {
        self.env = env
        self.mode = mode
    }

    public init(env: any Env) {
        self.init(env: env, mode: .nextStep)
    }

    public mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        needsReset = false
        return try env.reset(seed: seed, options: options)
    }

    public mutating func step(_ action: MLXArray) throws -> Step {
        if needsReset && mode == .nextStep {
            let resetResult = try env.reset(seed: nil, options: nil)
            needsReset = false
            return Step(
                obs: resetResult.obs,
                reward: 0.0,
                terminated: false,
                truncated: false,
                info: resetResult.info
            )
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
            info["final_observation"] = sendableValue(result.obs)
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
        info["final_observation"] = sendableValue(terminalObs)
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

private func sendableValue(_ value: MLXArray) -> InfoValue {
    .sendable(MLXArrayBox(array: value))
}

