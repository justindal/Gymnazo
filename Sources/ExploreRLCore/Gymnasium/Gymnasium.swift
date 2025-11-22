//
// Gymnasium.swift
//

@MainActor
public struct Gymnasium {
    public private(set) static var registry: [String: EnvSpec] = [:]

    private static var factories: [String: AnyEnvFactory] = [:]
    private static var didStart = false

    private struct WrapperConfig {
        let applyPassiveChecker: Bool
        let enforceOrder: Bool
        let disableRenderOrderEnforcing: Bool
        let maxEpisodeSteps: Int?
        let recordEpisodeStatistics: Bool
        let recordBufferLength: Int
        let recordStatsKey: String
    }

    /// starts the Gymnasium environment registry with the default environments.
    public static func start() {
        guard !didStart else { return }
        didStart = true
        GymnasiumRegistrations().registerDefaultEnvironments()
    }

    private class AnyEnvFactory {
        @MainActor
        func make(
            kwargs: [String: Any],
            config: WrapperConfig
        ) -> any Env {
            fatalError("EnvFactory.make(kwargs:config:) must be overridden")
        }
    }

    private final class EnvFactory<EnvType: Env>: AnyEnvFactory {
        private let builder: ([String: Any]) -> EnvType

        init(builder: @escaping ([String: Any]) -> EnvType) {
            self.builder = builder
        }

        @MainActor
        override func make(
            kwargs: [String: Any],
            config: WrapperConfig
        ) -> any Env {
            let env = builder(kwargs)
            return Gymnasium.applyDefaultWrappers(env: env, config: config)
        }
    }

    public static func register<EnvType: Env>(
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

        Self.registry[id] = spec
        Self.factories[id] = EnvFactory(builder: entryPoint)
    }

    public static func make(
        _ id: String,
        maxEpisodeSteps: Int? = nil,
        disableEnvChecker: Bool? = nil,
        disableRenderOrderEnforcing: Bool = false,
        recordEpisodeStatistics: Bool = true,
        recordBufferLength: Int = 100,
        recordStatsKey: String = "episode",
        kwargs: [String: Any] = [:]
    ) -> any Env {

        guard let spec: EnvSpec = Self.registry[id] else {
            fatalError("No environment registered with id \(id)")
        }

        return Self.make(
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

    public static func make(
        _ spec: EnvSpec,
        maxEpisodeSteps: Int? = nil,
        disableEnvChecker: Bool? = nil,
        disableRenderOrderEnforcing: Bool = false,
        recordEpisodeStatistics: Bool = true,
        recordBufferLength: Int = 100,
        recordStatsKey: String = "episode",
        kwargs: [String: Any] = [:]
    ) -> any Env {

        guard let factory = Self.factories[spec.id] else {
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

        // apply any additional wrappers specified on the spec, in order.
        if !appliedSpec.additional_wrappers.isEmpty {
            for wrapper in appliedSpec.additional_wrappers {
                env = wrapper.entryPoint(env, wrapper.kwargs)
                // propagate updated spec to the newly wrapped environment.
                env.spec = appliedSpec
            }
        }

        return env
    }

    private static func applyDefaultWrappers<E: Env>(
        env: E,
        config: WrapperConfig
    ) -> any Env {
        applyPassiveChecker(env: env, config: config)
    }

    private static func applyPassiveChecker<E: Env>(
        env: E,
        config: WrapperConfig
    ) -> any Env {
        if config.applyPassiveChecker {
            let wrapped = PassiveEnvChecker(env: env)
            return applyOrderEnforcing(env: wrapped, config: config)
        }
        return applyOrderEnforcing(env: env, config: config)
    }

    private static func applyOrderEnforcing<E: Env>(
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

    private static func applyTimeLimit<E: Env>(
        env: E,
        config: WrapperConfig
    ) -> any Env {
        guard let steps = config.maxEpisodeSteps else {
            return applyRecordEpisodeStatistics(env: env, config: config)
        }

        let wrapped = TimeLimit(env: env, maxEpisodeSteps: steps)
        return applyRecordEpisodeStatistics(env: wrapped, config: config)
    }

    private static func applyRecordEpisodeStatistics<E: Env>(
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
}