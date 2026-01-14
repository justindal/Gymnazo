//
//  Env.swift
//

public protocol Env<Observation, Action> {
    associatedtype Observation
    associatedtype Action

    associatedtype ObservationSpace: Space where ObservationSpace.T == Observation
    associatedtype ActionSpace: Space where ActionSpace.T == Action

    var action_space: ActionSpace { get }
    var observation_space: ObservationSpace { get }
    var spec: EnvSpec? { get set }
    var render_mode: String? { get set }

    var unwrapped: any Env<Observation, Action> { get }

    mutating func step(_ action: Action) -> Step<Observation>

    /// resets the environment to an initial state, returning an initial observation and info.
    /// this generates a new starting state, often with some randomness controlled by the optional seed parameter.
    mutating func reset(
        seed: UInt64?,
        options: [String: Any]?
    ) -> Reset<Observation>

    /// renders the environment.
    /// the return value depends on the `render_mode`.
    /// - "ansi": returns a `String`
    /// - "rgb_array": returns a `CGImage` (or platform equivalent)
    /// - "human": returns `nil`
    @discardableResult
    func render() -> Any?
}

public struct Step<Observation> {
    public var obs: Observation
    public var reward: Double
    public var terminated: Bool
    public var truncated: Bool
    public var info: Info
    public var final: EpisodeFinal<Observation>?

    public init(
        obs: Observation,
        reward: Double,
        terminated: Bool,
        truncated: Bool,
        info: Info = Info(),
        final: EpisodeFinal<Observation>? = nil
    ) {
        self.obs = obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
        self.final = final
    }
}

/// Auto-reset transition information.
public enum AutoResetTransition<Observation> {
    case none
    case willResetOnNextStep
    case didReset(observation: Observation, info: Info)
}

/// Episode end information.
public struct EpisodeFinal<Observation> {
    public var terminalObservation: Observation
    public var terminalInfo: Info
    public var autoReset: AutoResetTransition<Observation>

    public init(
        terminalObservation: Observation,
        terminalInfo: Info,
        autoReset: AutoResetTransition<Observation> = .none
    ) {
        self.terminalObservation = terminalObservation
        self.terminalInfo = terminalInfo
        self.autoReset = autoReset
    }
}

public struct Reset<Observation> {
    public var obs: Observation
    public var info: Info

    public init(obs: Observation, info: Info = Info()) {
        self.obs = obs
        self.info = info
    }
}

extension Env {
    /// override for `reset`
    public mutating func reset(seed: UInt64) -> Reset<Observation> {
        return self.reset(seed: seed, options: nil)
    }

    /// override for `reset` with no seed
    public mutating func reset() -> Reset<Observation> {
        return self.reset(seed: nil, options: nil)
    }

    public mutating func resetAny(seed: UInt64?, options: [String: Any]? = nil) -> Reset<Any> {
        let result = reset(seed: seed, options: options)
        return Reset(obs: result.obs as Any, info: result.info)
    }

    public mutating func stepAny(_ action: Any) -> Step<Any> {
        guard let typedAction = action as? Action else {
            fatalError(
                "Invalid action type \(type(of: action)) for environment action type \(Action.self)"
            )
        }

        let result = step(typedAction)

        let anyFinal: EpisodeFinal<Any>? = result.final.map { f in
            let anyAutoReset: AutoResetTransition<Any>
            switch f.autoReset {
            case .none:
                anyAutoReset = .none
            case .willResetOnNextStep:
                anyAutoReset = .willResetOnNextStep
            case .didReset(let obs, let info):
                anyAutoReset = .didReset(observation: obs as Any, info: info)
            }
            return EpisodeFinal(
                terminalObservation: f.terminalObservation as Any,
                terminalInfo: f.terminalInfo,
                autoReset: anyAutoReset
            )
        }

        return Step(
            obs: result.obs as Any,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info,
            final: anyFinal
        )
    }

    public var unwrapped: any Env<Observation, Action> {
        self
    }

    @discardableResult
    public func render() -> Any? {
        return nil
    }

    public func close() {
        // do nothing
    }
}
