//
// Gymnazo.swift
//

/// The registry of all available environments.
@MainActor
public private(set) var registry: [String: EnvSpec] = [:]


@MainActor
private var factories: [String: AnyEnvFactory] = [:]

@MainActor
private var didInitialize = false

/// Ensures the default environments are registered, called automatically on use.
@MainActor
private func ensureInitialized() {
    guard !didInitialize else { return }
    didInitialize = true
    GymnazoRegistrations().registerDefaultEnvironments()
}

private struct WrapperConfig {
    let applyPassiveChecker: Bool
    let enforceOrder: Bool
    let disableRenderOrderEnforcing: Bool
    let maxEpisodeSteps: Int?
    let recordEpisodeStatistics: Bool
    let recordBufferLength: Int
    let recordStatsKey: String
}

@MainActor
private class AnyEnvFactory {
    func make(
        kwargs: [String: Any],
        config: WrapperConfig
    ) -> any Env {
        fatalError("AnyEnvFactory.make(kwargs:config:) must be overridden")
    }
}

@MainActor
private final class EnvFactory<EnvType: Env>: AnyEnvFactory {
    private let builder: ([String: Any]) -> EnvType

    init(builder: @escaping ([String: Any]) -> EnvType) {
        self.builder = builder
    }

    override func make(
        kwargs: [String: Any],
        config: WrapperConfig
    ) -> any Env {
        let env = builder(kwargs)
        return applyDefaultWrappers(env: env, config: config)
    }
}


/// Registers a new environment with the given ID and entry point.
///
/// - Parameters:
///   - id: The unique identifier for the environment (e.g., "CartPole-v1").
///   - entryPoint: A closure that creates the environment from kwargs.
///   - maxEpisodeSteps: Optional maximum steps per episode.
///   - rewardThreshold: Optional reward threshold for solving the environment.
///   - nondeterministic: Whether the environment is nondeterministic.
@MainActor
public func register<EnvType: Env>(
    id: String,
    entryPoint: @escaping ([String: Any]) -> EnvType,
    maxEpisodeSteps: Int? = nil,
    rewardThreshold: Double? = nil,
    nondeterministic: Bool = false
) {
    let erasedEntryPoint: EnvCreator = { kwargs in entryPoint(kwargs) }

    let spec = EnvSpec(
        id: id,
        entry_point: .creator(erasedEntryPoint),
        rewardThreshold: rewardThreshold,
        nondeterministic: nondeterministic,
        maxEpisodeSteps: maxEpisodeSteps
    )

    registry[id] = spec
    factories[id] = EnvFactory(builder: entryPoint)
}

/// Creates an environment instance from the registry.
///
/// - Parameters:
///   - id: The environment ID (e.g., "CartPole-v1", "FrozenLake-v1").
///   - maxEpisodeSteps: Override the default max steps per episode.
///   - disableEnvChecker: Disable the passive environment checker.
///   - disableRenderOrderEnforcing: Disable render order enforcement.
///   - recordEpisodeStatistics: Whether to record episode statistics.
///   - recordBufferLength: Buffer length for statistics recording.
///   - recordStatsKey: Key for statistics in the info dict.
///   - kwargs: Additional keyword arguments passed to the environment.
/// - Returns: The created environment instance.
@MainActor
public func make(
    _ id: String,
    maxEpisodeSteps: Int? = nil,
    disableEnvChecker: Bool? = nil,
    disableRenderOrderEnforcing: Bool = false,
    recordEpisodeStatistics: Bool = true,
    recordBufferLength: Int = 100,
    recordStatsKey: String = "episode",
    kwargs: [String: Any] = [:]
) -> any Env {
    ensureInitialized()

    guard let spec: EnvSpec = registry[id] else {
        fatalError("No environment registered with id \(id). Available: \(Array(registry.keys))")
    }

    return make(
        spec,
        maxEpisodeSteps: maxEpisodeSteps,
        disableEnvChecker: disableEnvChecker,
        disableRenderOrderEnforcing: disableRenderOrderEnforcing,
        recordEpisodeStatistics: recordEpisodeStatistics,
        recordBufferLength: recordBufferLength,
        recordStatsKey: recordStatsKey,
        kwargs: kwargs
    )
}

/// Creates an environment instance from an EnvSpec.
///
/// - Parameters:
///   - spec: The environment specification.
///   - maxEpisodeSteps: Override the default max steps per episode.
///   - disableEnvChecker: Disable the passive environment checker.
///   - disableRenderOrderEnforcing: Disable render order enforcement.
///   - recordEpisodeStatistics: Whether to record episode statistics.
///   - recordBufferLength: Buffer length for statistics recording.
///   - recordStatsKey: Key for statistics in the info dict.
///   - kwargs: Additional keyword arguments passed to the environment.
/// - Returns: The created environment instance.
@MainActor
public func make(
    _ spec: EnvSpec,
    maxEpisodeSteps: Int? = nil,
    disableEnvChecker: Bool? = nil,
    disableRenderOrderEnforcing: Bool = false,
    recordEpisodeStatistics: Bool = true,
    recordBufferLength: Int = 100,
    recordStatsKey: String = "episode",
    kwargs: [String: Any] = [:]
) -> any Env {
    ensureInitialized()

    guard let factory = factories[spec.id] else {
        fatalError("No factory registered for id \(spec.id)")
    }

    guard let entryPoint = spec.entry_point else {
        fatalError("\(spec.id) registered but entry_point is not specified.")
    }

    switch entryPoint {
    case .string(_):
        fatalError("String entry points are not supported yet.")
    case .creator:
        break
    }

    precondition(recordBufferLength > 0, "recordBufferLength must be positive")

    var finalKwargs: [String: Any] = spec.kwargs
    finalKwargs.merge(kwargs) { (makeArg, _) in makeArg }

    let resolvedDisableChecker = disableEnvChecker ?? spec.disable_env_checker
    let resolvedMaxSteps = maxEpisodeSteps ?? spec.maxEpisodeSteps

    let config = WrapperConfig(
        applyPassiveChecker: !resolvedDisableChecker,
        enforceOrder: spec.order_enforce,
        disableRenderOrderEnforcing: disableRenderOrderEnforcing,
        maxEpisodeSteps: resolvedMaxSteps,
        recordEpisodeStatistics: recordEpisodeStatistics,
        recordBufferLength: recordBufferLength,
        recordStatsKey: recordStatsKey
    )

    var env = factory.make(
        kwargs: finalKwargs,
        config: config
    )

    var appliedSpec = spec
    appliedSpec.maxEpisodeSteps = resolvedMaxSteps
    appliedSpec.disable_env_checker = resolvedDisableChecker
    appliedSpec.kwargs = finalKwargs

    env.spec = appliedSpec

    // Apply any additional wrappers specified on the spec, in order.
    if !appliedSpec.additional_wrappers.isEmpty {
        for wrapper in appliedSpec.additional_wrappers {
            env = wrapper.entryPoint(env, wrapper.kwargs)
            env.spec = appliedSpec
        }
    }

    return env
}


@MainActor
private func applyDefaultWrappers<E: Env>(
    env: E,
    config: WrapperConfig
) -> any Env {
    applyPassiveChecker(env: env, config: config)
}

@MainActor
private func applyPassiveChecker<E: Env>(
    env: E,
    config: WrapperConfig
) -> any Env {
    if config.applyPassiveChecker {
        let wrapped = PassiveEnvChecker(env: env)
        return applyOrderEnforcing(env: wrapped, config: config)
    }
    return applyOrderEnforcing(env: env, config: config)
}

@MainActor
private func applyOrderEnforcing<E: Env>(
    env: E,
    config: WrapperConfig
) -> any Env {
    if config.enforceOrder {
        let wrapped = OrderEnforcing(
            env: env,
            disableRenderOrderEnforcing: config.disableRenderOrderEnforcing
        )
        return applyTimeLimit(env: wrapped, config: config)
    }
    return applyTimeLimit(env: env, config: config)
}

@MainActor
private func applyTimeLimit<E: Env>(
    env: E,
    config: WrapperConfig
) -> any Env {
    guard let steps = config.maxEpisodeSteps else {
        return applyRecordEpisodeStatistics(env: env, config: config)
    }

    let wrapped = TimeLimit(env: env, maxEpisodeSteps: steps)
    return applyRecordEpisodeStatistics(env: wrapped, config: config)
}

@MainActor
private func applyRecordEpisodeStatistics<E: Env>(
    env: E,
    config: WrapperConfig
) -> any Env {
    guard config.recordEpisodeStatistics else {
        return env
    }

    return RecordEpisodeStatistics(
        env: env,
        bufferLength: config.recordBufferLength,
        statsKey: config.recordStatsKey
    )
}
