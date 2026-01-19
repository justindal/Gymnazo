//
// Gymnazo.swift
// https://docs.swift.org/compiler/documentation/diagnostics/sendable-metatypes/
//

struct WrapperConfig: Sendable {
    let applyPassiveChecker: Bool
    let enforceOrder: Bool
    let disableRenderOrderEnforcing: Bool
    let maxEpisodeSteps: Int?
    let recordEpisodeStatistics: Bool
    let recordBufferLength: Int
    let recordStatsKey: String
}

public actor GymnazoRegistry {
    public static let shared = GymnazoRegistry()

    private var registry: [String: EnvSpec] = [:]
    private var didInitialize = false

    public init() {}

    /// The registry of all available environments.
    public func envSpecs() -> [String: EnvSpec] {
        ensureInitialized()
        return registry
    }

    /// The registered environment identifiers.
    public func ids() -> [String] {
        ensureInitialized()
        return registry.keys.sorted()
    }

    /// Registers a new environment with the given ID and entry point.
    ///
    /// - Parameters:
    ///   - id: The unique identifier for the environment (e.g., "CartPole").
    ///   - entryPoint: A closure that creates the environment from options.
    ///   - maxEpisodeSteps: Optional maximum steps per episode.
    ///   - rewardThreshold: Optional reward threshold for solving the environment.
    ///   - nondeterministic: Whether the environment is nondeterministic.
    public func register(
        id: String,
        entryPoint: @escaping @Sendable (EnvOptions) throws -> any Env,
        maxEpisodeSteps: Int? = nil,
        rewardThreshold: Double? = nil,
        nondeterministic: Bool = false
    ) {
        let spec = EnvSpec(
            id: id,
            entryPoint: entryPoint,
            rewardThreshold: rewardThreshold,
            nondeterministic: nondeterministic,
            maxEpisodeSteps: maxEpisodeSteps
        )
        registry[id] = spec
    }

    /// Returns true if an environment is registered for the given ID.
    public func isRegistered(_ id: String) -> Bool {
        ensureInitialized()
        return registry[id] != nil
    }

    func ensureInitialized() {
        guard !didInitialize else { return }
        didInitialize = true
        registerDefaultEnvironments()
    }

    func spec(for id: String) -> EnvSpec? {
        registry[id]
    }

}

/// Gymnazo API entry points.
public enum Gymnazo {
    /// The registry of all available environments.
    public static func registry() async -> [String: EnvSpec] {
        await GymnazoRegistry.shared.envSpecs()
    }

    /// The registered environment identifiers.
    public static func ids() async -> [String] {
        await GymnazoRegistry.shared.ids()
    }

    /// Returns true if an environment is registered for the given ID.
    public static func isRegistered(_ id: String) async -> Bool {
        await GymnazoRegistry.shared.isRegistered(id)
    }

    /// Registers a new environment with the given ID and entry point.
    ///
    /// - Parameters:
    ///   - id: The unique identifier for the environment (e.g., "CartPole").
    ///   - entryPoint: A closure that creates the environment from options.
    ///   - maxEpisodeSteps: Optional maximum steps per episode.
    ///   - rewardThreshold: Optional reward threshold for solving the environment.
    ///   - nondeterministic: Whether the environment is nondeterministic.
    public static func register(
        id: String,
        entryPoint: @escaping @Sendable (EnvOptions) throws -> any Env,
        maxEpisodeSteps: Int? = nil,
        rewardThreshold: Double? = nil,
        nondeterministic: Bool = false
    ) async {
        await GymnazoRegistry.shared.register(
            id: id,
            entryPoint: entryPoint,
            maxEpisodeSteps: maxEpisodeSteps,
            rewardThreshold: rewardThreshold,
            nondeterministic: nondeterministic
        )
    }

    /// Creates an environment instance from the registry.
    ///
    /// - Parameters:
    ///   - id: The environment ID (e.g., "CartPole", "FrozenLake").
    ///   - maxEpisodeSteps: Override the default max steps per episode.
    ///   - disableEnvChecker: Disable the passive environment checker.
    ///   - disableRenderOrderEnforcing: Disable render order enforcement.
    ///   - recordEpisodeStatistics: Whether to record episode statistics.
    ///   - recordBufferLength: Buffer length for statistics recording.
    /// Creates a typed environment instance from the registry.
    public static func make<Observation, Action>(
        _ id: String,
        maxEpisodeSteps: Int? = nil,
        disableEnvChecker: Bool? = nil,
        disableRenderOrderEnforcing: Bool = false,
        recordEpisodeStatistics: Bool = false,
        recordBufferLength: Int = 100,
        recordStatsKey: String = "episode",
        options: EnvOptions = [:]
    ) async throws -> AnyEnv<Observation, Action> {
        let env = try await GymnazoRegistry.shared.make(
            id,
            maxEpisodeSteps: maxEpisodeSteps,
            disableEnvChecker: disableEnvChecker,
            disableRenderOrderEnforcing: disableRenderOrderEnforcing,
            recordEpisodeStatistics: recordEpisodeStatistics,
            recordBufferLength: recordBufferLength,
            recordStatsKey: recordStatsKey,
            options: options
        )
        guard let typed = env as? any Env<Observation, Action> else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "Env<\(Observation.self), \(Action.self)>",
                actual: String(describing: type(of: env))
            )
        }
        return AnyEnv(typed)
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
    /// Creates a typed environment instance from an EnvSpec.
    public static func make<Observation, Action>(
        _ spec: EnvSpec,
        maxEpisodeSteps: Int? = nil,
        disableEnvChecker: Bool? = nil,
        disableRenderOrderEnforcing: Bool = false,
        recordEpisodeStatistics: Bool = false,
        recordBufferLength: Int = 100,
        recordStatsKey: String = "episode",
        options: EnvOptions = [:]
    ) async throws -> AnyEnv<Observation, Action> {
        let env = try await GymnazoRegistry.shared.make(
            spec,
            maxEpisodeSteps: maxEpisodeSteps,
            disableEnvChecker: disableEnvChecker,
            disableRenderOrderEnforcing: disableRenderOrderEnforcing,
            recordEpisodeStatistics: recordEpisodeStatistics,
            recordBufferLength: recordBufferLength,
            recordStatsKey: recordStatsKey,
            options: options
        )
        guard let typed = env as? any Env<Observation, Action> else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "Env<\(Observation.self), \(Action.self)>",
                actual: String(describing: type(of: env))
            )
        }
        return AnyEnv(typed)
    }

    /// Creates a vectorized environment with multiple sub-environments.
    ///
    /// This is the vector environment equivalent of `make()`. It creates a `SyncVectorEnv`
    /// that manages multiple instances of the specified environment.
    ///
    /// - Parameters:
    ///   - id: The environment ID (e.g., "CartPole", "FrozenLake").
    ///   - numEnvs: The number of sub-environments to create.
    ///   - maxEpisodeSteps: Override the default max steps per episode.
    ///   - disableEnvChecker: Disable the passive environment checker.
    ///   - disableRenderOrderEnforcing: Disable render order enforcement.
    ///   - recordEpisodeStatistics: Whether to record episode statistics.
    ///   - recordBufferLength: Buffer length for statistics recording.
    ///   - recordStatsKey: Key for statistics in the info dict.
    ///   - autoresetMode: The autoreset mode for the vector environment. Default is `.nextStep`.
    ///   - options: Additional options passed to each environment.
    /// - Returns: A `SyncVectorEnv` managing the created sub-environments.
    @MainActor
    public static func makeVec<Action>(
        _ id: String,
        numEnvs: Int,
        maxEpisodeSteps: Int? = nil,
        disableEnvChecker: Bool? = nil,
        disableRenderOrderEnforcing: Bool = false,
        recordEpisodeStatistics: Bool = false,
        recordBufferLength: Int = 100,
        recordStatsKey: String = "episode",
        autoresetMode: AutoresetMode = .nextStep,
        options: EnvOptions = [:]
    ) async throws -> SyncVectorEnv<Action> {
        try await GymnazoRegistry.shared.makeVec(
            id,
            numEnvs: numEnvs,
            maxEpisodeSteps: maxEpisodeSteps,
            disableEnvChecker: disableEnvChecker,
            disableRenderOrderEnforcing: disableRenderOrderEnforcing,
            recordEpisodeStatistics: recordEpisodeStatistics,
            recordBufferLength: recordBufferLength,
            recordStatsKey: recordStatsKey,
            autoresetMode: autoresetMode,
            options: options
        )
    }

    /// Creates a vectorized environment from an array of environment factory functions.
    ///
    /// This provides more flexibility than `makeVec(_:numEnvs:)` by allowing
    /// different configurations for each sub-environment.
    ///
    /// - Parameters:
    ///   - envFns: Array of closures that create environments.
    ///   - autoresetMode: The autoreset mode for the vector environment. Default is `.nextStep`.
    /// - Returns: A `SyncVectorEnv` managing the created sub-environments.
    @MainActor
    public static func makeVec<Action>(
        envFns: [() -> any Env],
        autoresetMode: AutoresetMode = .nextStep
    ) async throws -> SyncVectorEnv<Action> {
        try await GymnazoRegistry.shared.makeVec(
            envFns: envFns,
            autoresetMode: autoresetMode
        )
    }

    /// Creates a vectorized environment with multiple sub-environments.
    ///
    /// This is an enhanced version of `makeVec` that supports both synchronous and
    /// asynchronous vectorization modes.
    ///
    /// - Parameters:
    ///   - id: The environment ID (e.g., "CartPole", "FrozenLake").
    ///   - numEnvs: The number of sub-environments to create.
    ///   - vectorizationMode: The vectorization mode (`.sync` or `.async`). Default is `.sync`.
    ///   - maxEpisodeSteps: Override the default max steps per episode.
    ///   - disableEnvChecker: Disable the passive environment checker.
    ///   - disableRenderOrderEnforcing: Disable render order enforcement.
    ///   - recordEpisodeStatistics: Whether to record episode statistics.
    ///   - recordBufferLength: Buffer length for statistics recording.
    ///   - recordStatsKey: Key for statistics in the info dict.
    ///   - autoresetMode: The autoreset mode for the vector environment. Default is `.nextStep`.
    ///   - options: Additional options passed to each environment.
    /// - Returns: A `VectorEnv` managing the created sub-environments.
    @MainActor
    public static func makeVec<Action>(
        _ id: String,
        numEnvs: Int,
        vectorizationMode: VectorizationMode = .sync,
        maxEpisodeSteps: Int? = nil,
        disableEnvChecker: Bool? = nil,
        disableRenderOrderEnforcing: Bool = false,
        recordEpisodeStatistics: Bool = false,
        recordBufferLength: Int = 100,
        recordStatsKey: String = "episode",
        autoresetMode: AutoresetMode = .nextStep,
        options: EnvOptions = [:]
    ) async throws -> any VectorEnv<Action> {
        try await GymnazoRegistry.shared.makeVec(
            id,
            numEnvs: numEnvs,
            vectorizationMode: vectorizationMode,
            maxEpisodeSteps: maxEpisodeSteps,
            disableEnvChecker: disableEnvChecker,
            disableRenderOrderEnforcing: disableRenderOrderEnforcing,
            recordEpisodeStatistics: recordEpisodeStatistics,
            recordBufferLength: recordBufferLength,
            recordStatsKey: recordStatsKey,
            autoresetMode: autoresetMode,
            options: options
        )
    }

    /// Creates an asynchronous vectorized environment with multiple sub-environments.
    ///
    /// This creates an `AsyncVectorEnv` that can run environment operations in parallel
    /// using Swift Concurrency.
    ///
    /// - Parameters:
    ///   - id: The environment ID (e.g., "CartPole", "FrozenLake").
    ///   - numEnvs: The number of sub-environments to create.
    ///   - maxEpisodeSteps: Override the default max steps per episode.
    ///   - disableEnvChecker: Disable the passive environment checker.
    ///   - disableRenderOrderEnforcing: Disable render order enforcement.
    ///   - recordEpisodeStatistics: Whether to record episode statistics.
    ///   - recordBufferLength: Buffer length for statistics recording.
    ///   - recordStatsKey: Key for statistics in the info dict.
    ///   - autoresetMode: The autoreset mode for the vector environment. Default is `.nextStep`.
    ///   - options: Additional options passed to each environment.
    /// - Returns: An `AsyncVectorEnv` managing the created sub-environments.
    @MainActor
    public static func makeVecAsync<Action>(
        _ id: String,
        numEnvs: Int,
        maxEpisodeSteps: Int? = nil,
        disableEnvChecker: Bool? = nil,
        disableRenderOrderEnforcing: Bool = false,
        recordEpisodeStatistics: Bool = false,
        recordBufferLength: Int = 100,
        recordStatsKey: String = "episode",
        autoresetMode: AutoresetMode = .nextStep,
        options: EnvOptions = [:]
    ) async throws -> AsyncVectorEnv<Action> {
        try await GymnazoRegistry.shared.makeVecAsync(
            id,
            numEnvs: numEnvs,
            maxEpisodeSteps: maxEpisodeSteps,
            disableEnvChecker: disableEnvChecker,
            disableRenderOrderEnforcing: disableRenderOrderEnforcing,
            recordEpisodeStatistics: recordEpisodeStatistics,
            recordBufferLength: recordBufferLength,
            recordStatsKey: recordStatsKey,
            autoresetMode: autoresetMode,
            options: options
        )
    }

    /// Creates an asynchronous vectorized environment from an array of environment factory functions.
    ///
    /// This provides more flexibility than `makeVecAsync(_:numEnvs:)` by allowing
    /// different configurations for each sub-environment.
    ///
    /// - Parameters:
    ///   - envFns: Array of closures that create environments.
    ///   - autoresetMode: The autoreset mode for the vector environment. Default is `.nextStep`.
    /// - Returns: An `AsyncVectorEnv` managing the created sub-environments.
    @MainActor
    public static func makeVecAsync<Action>(
        envFns: [@Sendable () -> any Env],
        autoresetMode: AutoresetMode = .nextStep
    ) async throws -> AsyncVectorEnv<Action> {
        try await GymnazoRegistry.shared.makeVecAsync(
            envFns: envFns,
            autoresetMode: autoresetMode
        )
    }
}

/// Vectorization mode for vector environments.
public enum VectorizationMode: String, Sendable {
    /// Synchronous execution - environments run sequentially.
    case sync
    /// Asynchronous execution - environments run in parallel using Swift Concurrency.
    case async
}

extension GymnazoRegistry {
    /// Creates an environment instance from the registry.
    ///
    /// - Parameters:
    ///   - id: The environment ID (e.g., "CartPole", "FrozenLake").
    ///   - maxEpisodeSteps: Override the default max steps per episode.
    ///   - disableEnvChecker: Disable the passive environment checker.
    ///   - disableRenderOrderEnforcing: Disable render order enforcement.
    ///   - recordEpisodeStatistics: Whether to record episode statistics.
    ///   - recordBufferLength: Buffer length for statistics recording.
    ///   - recordStatsKey: Key for statistics in the info dict.
    ///   - options: Additional options passed to the environment.
    /// - Returns: The created environment instance.
    public nonisolated func make(
        _ id: String,
        maxEpisodeSteps: Int? = nil,
        disableEnvChecker: Bool? = nil,
        disableRenderOrderEnforcing: Bool = false,
        recordEpisodeStatistics: Bool = false,
        recordBufferLength: Int = 100,
        recordStatsKey: String = "episode",
        options: EnvOptions = [:]
    ) async throws -> any Env {
        await ensureInitialized()
        guard let spec = await spec(for: id) else {
            throw GymnazoError.unregisteredEnvironment(id: id)
        }
        return try await make(
            spec,
            maxEpisodeSteps: maxEpisodeSteps,
            disableEnvChecker: disableEnvChecker,
            disableRenderOrderEnforcing: disableRenderOrderEnforcing,
            recordEpisodeStatistics: recordEpisodeStatistics,
            recordBufferLength: recordBufferLength,
            recordStatsKey: recordStatsKey,
            options: options
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
    ///   - options: Additional options passed to the environment.
    /// - Returns: The created environment instance.
    public nonisolated func make(
        _ spec: EnvSpec,
        maxEpisodeSteps: Int? = nil,
        disableEnvChecker: Bool? = nil,
        disableRenderOrderEnforcing: Bool = false,
        recordEpisodeStatistics: Bool = false,
        recordBufferLength: Int = 100,
        recordStatsKey: String = "episode",
        options: EnvOptions = [:]
    ) async throws -> any Env {
        await ensureInitialized()
        return try buildEnv(
            spec: spec,
            maxEpisodeSteps: maxEpisodeSteps,
            disableEnvChecker: disableEnvChecker,
            disableRenderOrderEnforcing: disableRenderOrderEnforcing,
            recordEpisodeStatistics: recordEpisodeStatistics,
            recordBufferLength: recordBufferLength,
            recordStatsKey: recordStatsKey,
            options: options
        )
    }

    /// Creates a vectorized environment with multiple sub-environments.
    ///
    /// This is the vector environment equivalent of `make()`. It creates a `SyncVectorEnv`
    /// that manages multiple instances of the specified environment.
    ///
    /// - Parameters:
    ///   - id: The environment ID (e.g., "CartPole", "FrozenLake").
    ///   - numEnvs: The number of sub-environments to create.
    ///   - maxEpisodeSteps: Override the default max steps per episode.
    ///   - disableEnvChecker: Disable the passive environment checker.
    ///   - disableRenderOrderEnforcing: Disable render order enforcement.
    ///   - recordEpisodeStatistics: Whether to record episode statistics.
    ///   - recordBufferLength: Buffer length for statistics recording.
    ///   - recordStatsKey: Key for statistics in the info dict.
    ///   - autoresetMode: The autoreset mode for the vector environment. Default is `.nextStep`.
    ///   - options: Additional options passed to each environment.
    /// - Returns: A `SyncVectorEnv` managing the created sub-environments.
    @MainActor
    public func makeVec<Action>(
        _ id: String,
        numEnvs: Int,
        maxEpisodeSteps: Int? = nil,
        disableEnvChecker: Bool? = nil,
        disableRenderOrderEnforcing: Bool = false,
        recordEpisodeStatistics: Bool = false,
        recordBufferLength: Int = 100,
        recordStatsKey: String = "episode",
        autoresetMode: AutoresetMode = .nextStep,
        options: EnvOptions = [:]
    ) async throws -> SyncVectorEnv<Action> {
        guard numEnvs > 0 else {
            throw GymnazoError.invalidNumEnvs(numEnvs)
        }
        await ensureInitialized()
        guard let spec = await spec(for: id) else {
            throw GymnazoError.unregisteredEnvironment(id: id)
        }

        let envs: [any Env] = try (0..<numEnvs).map { _ in
            try buildEnv(
                spec: spec,
                maxEpisodeSteps: maxEpisodeSteps,
                disableEnvChecker: disableEnvChecker,
                disableRenderOrderEnforcing: disableRenderOrderEnforcing,
                recordEpisodeStatistics: recordEpisodeStatistics,
                recordBufferLength: recordBufferLength,
                recordStatsKey: recordStatsKey,
                options: options
            )
        }

        let vectorEnv = try SyncVectorEnv<Action>(
            envs: envs,
            copyObservations: true,
            autoresetMode: autoresetMode
        )

        vectorEnv.spec = spec

        return vectorEnv
    }

    /// Creates a vectorized environment from an array of environment factory functions.
    ///
    /// This provides more flexibility than `makeVec(_:numEnvs:)` by allowing
    /// different configurations for each sub-environment.
    ///
    /// - Parameters:
    ///   - envFns: Array of closures that create environments.
    ///   - autoresetMode: The autoreset mode for the vector environment. Default is `.nextStep`.
    /// - Returns: A `SyncVectorEnv` managing the created sub-environments.
    @MainActor
    public func makeVec<Action>(
        envFns: [() -> any Env],
        autoresetMode: AutoresetMode = .nextStep
    ) async throws -> SyncVectorEnv<Action> {
        try SyncVectorEnv(
            envFns: envFns,
            copyObservations: true,
            autoresetMode: autoresetMode
        )
    }

    /// Creates a vectorized environment with multiple sub-environments.
    ///
    /// This is an enhanced version of `makeVec` that supports both synchronous and
    /// asynchronous vectorization modes.
    ///
    /// - Parameters:
    ///   - id: The environment ID (e.g., "CartPole", "FrozenLake").
    ///   - numEnvs: The number of sub-environments to create.
    ///   - vectorizationMode: The vectorization mode (`.sync` or `.async`). Default is `.sync`.
    ///   - maxEpisodeSteps: Override the default max steps per episode.
    ///   - disableEnvChecker: Disable the passive environment checker.
    ///   - disableRenderOrderEnforcing: Disable render order enforcement.
    ///   - recordEpisodeStatistics: Whether to record episode statistics.
    ///   - recordBufferLength: Buffer length for statistics recording.
    ///   - recordStatsKey: Key for statistics in the info dict.
    ///   - autoresetMode: The autoreset mode for the vector environment. Default is `.nextStep`.
    ///   - options: Additional options passed to each environment.
    /// - Returns: A `VectorEnv` managing the created sub-environments.
    @MainActor
    public func makeVec<Action>(
        _ id: String,
        numEnvs: Int,
        vectorizationMode: VectorizationMode = .sync,
        maxEpisodeSteps: Int? = nil,
        disableEnvChecker: Bool? = nil,
        disableRenderOrderEnforcing: Bool = false,
        recordEpisodeStatistics: Bool = false,
        recordBufferLength: Int = 100,
        recordStatsKey: String = "episode",
        autoresetMode: AutoresetMode = .nextStep,
        options: EnvOptions = [:]
    ) async throws -> any VectorEnv<Action> {
        switch vectorizationMode {
        case .sync:
            return try await makeVec(
                id,
                numEnvs: numEnvs,
                maxEpisodeSteps: maxEpisodeSteps,
                disableEnvChecker: disableEnvChecker,
                disableRenderOrderEnforcing: disableRenderOrderEnforcing,
                recordEpisodeStatistics: recordEpisodeStatistics,
                recordBufferLength: recordBufferLength,
                recordStatsKey: recordStatsKey,
                autoresetMode: autoresetMode,
                options: options
            )
        case .async:
            return try await makeVecAsync(
                id,
                numEnvs: numEnvs,
                maxEpisodeSteps: maxEpisodeSteps,
                disableEnvChecker: disableEnvChecker,
                disableRenderOrderEnforcing: disableRenderOrderEnforcing,
                recordEpisodeStatistics: recordEpisodeStatistics,
                recordBufferLength: recordBufferLength,
                recordStatsKey: recordStatsKey,
                autoresetMode: autoresetMode,
                options: options
            )
        }
    }

    /// Creates an asynchronous vectorized environment with multiple sub-environments.
    ///
    /// This creates an `AsyncVectorEnv` that can run environment operations in parallel
    /// using Swift Concurrency.
    ///
    /// - Parameters:
    ///   - id: The environment ID (e.g., "CartPole", "FrozenLake").
    ///   - numEnvs: The number of sub-environments to create.
    ///   - maxEpisodeSteps: Override the default max steps per episode.
    ///   - disableEnvChecker: Disable the passive environment checker.
    ///   - disableRenderOrderEnforcing: Disable render order enforcement.
    ///   - recordEpisodeStatistics: Whether to record episode statistics.
    ///   - recordBufferLength: Buffer length for statistics recording.
    ///   - recordStatsKey: Key for statistics in the info dict.
    ///   - autoresetMode: The autoreset mode for the vector environment. Default is `.nextStep`.
    ///   - options: Additional options passed to each environment.
    /// - Returns: An `AsyncVectorEnv` managing the created sub-environments.
    @MainActor
    public func makeVecAsync<Action>(
        _ id: String,
        numEnvs: Int,
        maxEpisodeSteps: Int? = nil,
        disableEnvChecker: Bool? = nil,
        disableRenderOrderEnforcing: Bool = false,
        recordEpisodeStatistics: Bool = false,
        recordBufferLength: Int = 100,
        recordStatsKey: String = "episode",
        autoresetMode: AutoresetMode = .nextStep,
        options: EnvOptions = [:]
    ) async throws -> AsyncVectorEnv<Action> {
        guard numEnvs > 0 else {
            throw GymnazoError.invalidNumEnvs(numEnvs)
        }
        await ensureInitialized()
        guard let spec = await spec(for: id) else {
            throw GymnazoError.unregisteredEnvironment(id: id)
        }

        let envs = try (0..<numEnvs).map { _ in
            try buildEnv(
                spec: spec,
                maxEpisodeSteps: maxEpisodeSteps,
                disableEnvChecker: disableEnvChecker,
                disableRenderOrderEnforcing: disableRenderOrderEnforcing,
                recordEpisodeStatistics: recordEpisodeStatistics,
                recordBufferLength: recordBufferLength,
                recordStatsKey: recordStatsKey,
                options: options
            )
        }

        let vectorEnv = try AsyncVectorEnv<Action>(
            envs: envs,
            copyObservations: true,
            autoresetMode: autoresetMode
        )

        vectorEnv.spec = spec

        return vectorEnv
    }

    /// Creates an asynchronous vectorized environment from an array of environment factory functions.
    ///
    /// This provides more flexibility than `makeVecAsync(_:numEnvs:)` by allowing
    /// different configurations for each sub-environment.
    ///
    /// - Parameters:
    ///   - envFns: Array of closures that create environments.
    ///   - autoresetMode: The autoreset mode for the vector environment. Default is `.nextStep`.
    /// - Returns: An `AsyncVectorEnv` managing the created sub-environments.
    @MainActor
    public func makeVecAsync<Action>(
        envFns: [@Sendable () -> any Env],
        autoresetMode: AutoresetMode = .nextStep
    ) async throws -> AsyncVectorEnv<Action> {
        try AsyncVectorEnv(
            envFns: envFns,
            copyObservations: true,
            autoresetMode: autoresetMode
        )
    }
}

private func buildEnv(
    spec: EnvSpec,
    maxEpisodeSteps: Int?,
    disableEnvChecker: Bool?,
    disableRenderOrderEnforcing: Bool,
    recordEpisodeStatistics: Bool,
    recordBufferLength: Int,
    recordStatsKey: String,
    options: EnvOptions
) throws -> any Env {
    guard recordBufferLength > 0 else {
        throw GymnazoError.invalidRecordBufferLength(recordBufferLength)
    }

    var finalOptions = spec.options
    finalOptions.storage.merge(options.storage) { _, new in new }

    let resolvedDisableChecker = disableEnvChecker ?? spec.disableEnvChecker
    let resolvedMaxSteps = maxEpisodeSteps ?? spec.maxEpisodeSteps

    let config = WrapperConfig(
        applyPassiveChecker: !resolvedDisableChecker,
        enforceOrder: spec.orderEnforce,
        disableRenderOrderEnforcing: disableRenderOrderEnforcing,
        maxEpisodeSteps: resolvedMaxSteps,
        recordEpisodeStatistics: recordEpisodeStatistics,
        recordBufferLength: recordBufferLength,
        recordStatsKey: recordStatsKey
    )

    var env = try applyDefaultWrappers(
        env: try spec.entryPoint(finalOptions),
        config: config
    )

    var appliedSpec = spec
    appliedSpec.maxEpisodeSteps = resolvedMaxSteps
    appliedSpec.disableEnvChecker = resolvedDisableChecker
    appliedSpec.options = finalOptions

    env.spec = appliedSpec

    if !appliedSpec.additionalWrappers.isEmpty {
        for wrapper in appliedSpec.additionalWrappers {
            env = try wrapper.entryPoint(env, wrapper.options)
            env.spec = appliedSpec
        }
    }

    return env
}

private func applyDefaultWrappers<E: Env>(
    env: E,
    config: WrapperConfig
) throws -> any Env {
    try applyPassiveChecker(env: env, config: config)
}

private func applyPassiveChecker<E: Env>(
    env: E,
    config: WrapperConfig
) throws -> any Env {
    if config.applyPassiveChecker {
        let wrapped = PassiveEnvChecker(env: env)
        return try applyOrderEnforcing(env: wrapped, config: config)
    }
    return try applyOrderEnforcing(env: env, config: config)
}

private func applyOrderEnforcing<E: Env>(
    env: E,
    config: WrapperConfig
) throws -> any Env {
    if config.enforceOrder {
        let wrapped = OrderEnforcing(
            env: env,
            disableRenderOrderEnforcing: config.disableRenderOrderEnforcing
        )
        return try applyTimeLimit(env: wrapped, config: config)
    }
    return try applyTimeLimit(env: env, config: config)
}

private func applyTimeLimit<E: Env>(
    env: E,
    config: WrapperConfig
) throws -> any Env {
    guard let steps = config.maxEpisodeSteps, steps != -1 else {
        return try applyRecordEpisodeStatistics(env: env, config: config)
    }

    let wrapped = try TimeLimit(env: env, maxEpisodeSteps: steps)
    return try applyRecordEpisodeStatistics(env: wrapped, config: config)
}

private func applyRecordEpisodeStatistics<E: Env>(
    env: E,
    config: WrapperConfig
) throws -> any Env {
    guard config.recordEpisodeStatistics else {
        return env
    }

    return try RecordEpisodeStatistics(
        env: env,
        bufferLength: config.recordBufferLength,
        statsKey: config.recordStatsKey
    )
}
