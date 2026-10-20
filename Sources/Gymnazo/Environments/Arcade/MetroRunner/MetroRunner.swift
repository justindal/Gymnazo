import MLX

/// A native Metro Run environment with five discrete actions and flattened state observations.
///
/// The MLX representation concatenates entities, entity_mask, and player in sorted key order.
/// The observation space is a Box of 2,089 floats suitable for standard MLP policies.
/// This simulator retains the supplied approximation's collision, pickup, and flight limitations.
public struct MetroRunner: Env {
    public let actionSpace: any Space
    public let observationSpace: any Space
    public var spec: EnvSpec?
    public var renderMode: RenderMode?
    private var environment: MetroRunnerValueEnvironment

    public init(renderMode: RenderMode? = nil, config: MetroRunnerConfig = MetroRunnerConfig()) throws {
        guard renderMode == nil || renderMode == .human || renderMode == .ansi else {
            throw GymnazoError.invalidConfiguration("MetroRunner supports human snapshots and ANSI rendering.")
        }
        self.environment = try MetroRunnerValueEnvironment(config: config)
        self.renderMode = renderMode
        self.actionSpace = Discrete(n: MetroRunnerAction.allCases.count)
        self.observationSpace = Box(low: -1, high: 1, shape: [MetroRunnerObservation.flatCount])
    }

    public mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        var typed: ValueResetOptions = [:]
        for (key, value) in options?.storage ?? [:] {
            switch value {
            case let value as String: typed[key] = .text(value)
            case let value as Int: typed[key] = .discrete(value)
            case let value as UInt64: typed[key] = .text(String(value))
            default: throw GymnazoError.invalidConfiguration("Invalid MetroRunner reset option: \(key).")
            }
        }
        let reset = try environment.resetValue(seed: seed, options: typed)
        return Reset(obs: MLXValueCodec.encode(try MetroRunnerObservation.flatten(reset.observation)),
                     info: Info(valueInfo: reset.info))
    }

    public mutating func step(_ action: MLXArray) throws -> Step {
        guard action.size == 1, action.dtype.isInteger else {
            throw GymnazoError.invalidAction("MetroRunner requires one integer action from 0 through 4.")
        }
        eval(action)
        let value = action.item(Int.self)
        let step = try environment.stepValue(value)
        return Step(obs: MLXValueCodec.encode(try MetroRunnerObservation.flatten(step.observation)),
                    reward: step.reward, terminated: step.terminated, truncated: step.truncated,
                    info: Info(valueInfo: step.info))
    }

    public var currentSnapshot: MetroRunnerSnapshot { environment.currentSnapshot }

    public func render() throws -> RenderOutput? {
        switch renderMode {
        case nil: return nil
        case .human: return .other(currentSnapshot)
        case .ansi:
            let s = currentSnapshot
            return .ansi("Metro Run | distance \(Int(s.distance)) | coins \(s.coins) | score \(s.score) | \(s.dead ? "crashed" : "running")")
        default: throw GymnazoError.invalidConfiguration("Unsupported MetroRunner render mode.")
        }
    }

    public mutating func close() { environment.close() }
}
