//
//  AsyncVectorEnv.swift
//

import MLX
import Foundation
import os

/// Sendable result type for step operations crossing actor boundaries.
public struct EnvStepResult: Sendable {
    public let index: Int
    public let observation: [Float]
    public let reward: Float
    public let terminated: Bool
    public let truncated: Bool
    public let info: Info

    public init(
        index: Int,
        observation: [Float],
        reward: Float,
        terminated: Bool,
        truncated: Bool,
        info: Info
    ) {
        self.index = index
        self.observation = observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
    }
}

/// Sendable result type for reset operations crossing actor boundaries.
public struct EnvResetResult: Sendable {
    public let index: Int
    public let observation: [Float]
    public let info: Info

    public init(index: Int, observation: [Float], info: Info) {
        self.index = index
        self.observation = observation
        self.info = info
    }
}

public struct UnsafeEnvBox: @unchecked Sendable {
    public var env: any Env
    public init(env: any Env) { self.env = env }
}

public struct UnsafeAction: @unchecked Sendable {
    public let value: MLXArray
    public init(_ value: MLXArray) { self.value = value }
}

/// Actor that wraps a single environment for isolated parallel execution.
public actor EnvironmentActor {
    private var needsReset: Bool = true
    private let index: Int
    private let autoresetMode: AutoresetMode
    private var env: any Env

    public init(index: Int, envBox: UnsafeEnvBox, autoresetMode: AutoresetMode) {
        self.index = index
        self.autoresetMode = autoresetMode
        self.env = envBox.env
    }

    public func step(_ action: UnsafeAction) throws -> EnvStepResult {
        if needsReset {
            if autoresetMode == .nextStep {
                let resetResult = try env.reset(seed: nil, options: nil)
                needsReset = false
                let obsArray: [Float] = resetResult.obs.asArray(Float.self)
                return EnvStepResult(
                    index: index,
                    observation: obsArray,
                    reward: 0.0,
                    terminated: false,
                    truncated: false,
                    info: resetResult.info
                )
            } else {
                throw GymnazoError.vectorEnvNeedsReset(index: index)
            }
        }

        let result = try env.step(action.value)
        let done = result.terminated || result.truncated
        let finalObsArray: [Float] = result.obs.asArray(Float.self)

        if done && autoresetMode == .sameStep {
            let terminalInfo = result.info
            let terminalObs = result.obs
            let resetResult = try env.reset(seed: nil, options: nil)
            needsReset = false
            let obsArray: [Float] = resetResult.obs.asArray(Float.self)
            var info = resetResult.info
            if let value = sendableValue(terminalObs) {
                info["final_observation"] = value
            }
            info["final_info"] = .object(terminalInfo.storage)
            return EnvStepResult(
                index: index,
                observation: obsArray,
                reward: Float(result.reward),
                terminated: result.terminated,
                truncated: result.truncated,
                info: info
            )
        }

        needsReset = done
        var info = result.info
        if done, let value = sendableValue(result.obs) {
            info["final_observation"] = value
        }
        if done {
            info["final_info"] = .object(result.info.storage)
        }
        return EnvStepResult(
            index: index,
            observation: finalObsArray,
            reward: Float(result.reward),
            terminated: result.terminated,
            truncated: result.truncated,
            info: info
        )
    }

    public func reset(seed: UInt64?, options: EnvOptions?) throws -> EnvResetResult {
        let result = try env.reset(seed: seed, options: options)
        needsReset = false
        let obsArray: [Float] = result.obs.asArray(Float.self)
        return EnvResetResult(index: index, observation: obsArray, info: result.info)
    }

    public func close() {
        env.close()
    }

    public func markNeedsReset(_ value: Bool) {
        needsReset = value
    }
}

/// Vectorized environment that runs multiple environments in parallel using Swift Actors.
///
/// `AsyncVectorEnv` manages multiple sub-environments using actor isolation, allowing
/// true parallel execution across multiple threads. Each environment runs in its own
/// actor, and `TaskGroup` coordinates parallel step and reset operations.
///
/// ## Example
///
/// ```swift
/// // Create vector environment with 4 CartPole instances
/// let envs = AsyncVectorEnv(envFns: [
///     { CartPole() },
///     { CartPole() },
///     { CartPole() },
///     { CartPole() }
/// ])
///
/// // Reset all environments in parallel
/// let (obs, _) = await envs.resetAsync(seed: 42)
/// // obs.shape == [4, 4] for 4 envs with 4-dimensional observations
///
/// // Step all environments in parallel
/// let result = await envs.stepAsync([action1, action2, action3, action4])
/// // result.observations.shape == [4, 4]
/// // result.rewards.shape == [4]
/// ```
///
/// ## Autoreset Behavior
///
/// By default, `AsyncVectorEnv` uses next-step autoreset. When a sub-environment
/// terminates or truncates:
/// 1. The final observation is stored in `infos["final_observation"]`
/// 2. The final info is stored in `infos["final_info"]`
/// 3. On the next step, the sub-environment is automatically reset
@MainActor
public final class AsyncVectorEnv: VectorEnv {

    public let numEnvs: Int

    private let actors: [EnvironmentActor]

    public let singleObservationSpace: any Space

    public let singleActionSpace: any Space

    public private(set) var observationSpace: any Space

    public private(set) var actionSpace: any Space

    public var spec: EnvSpec?

    public let renderMode: RenderMode?

    public let autoresetMode: AutoresetMode

    public private(set) var closed: Bool = false

    private let observationShape: [Int]

    private let copyObservations: Bool

    /// Creates a new `AsyncVectorEnv` from an array of environment factory functions.
    ///
    /// - Parameters:
    ///   - envFns: Array of closures that create environments.
    ///   - copyObservations: Whether to copy observations. Default is `true`.
    ///   - autoresetMode: The autoreset mode to use. Default is `.nextStep`.
    public init(
        envFns: [@Sendable () -> any Env],
        copyObservations: Bool = true,
        autoresetMode: AutoresetMode = .nextStep
    ) throws {
        guard !envFns.isEmpty else {
            throw GymnazoError.invalidNumEnvs(envFns.count)
        }
        let wrappedEnvs = envFns.map { $0() }
        let probeEnv = wrappedEnvs[0]
        self.numEnvs = envFns.count
        self.autoresetMode = autoresetMode
        self.copyObservations = copyObservations

        let actors = wrappedEnvs.enumerated().map { index, env in
            EnvironmentActor(index: index, envBox: UnsafeEnvBox(env: env), autoresetMode: autoresetMode)
        }
        self.actors = actors

        self.singleObservationSpace = probeEnv.observationSpace
        self.singleActionSpace = probeEnv.actionSpace
        self.renderMode = probeEnv.renderMode
        self.spec = probeEnv.spec
        self.observationShape = probeEnv.observationSpace.shape ?? [4]

        self.observationSpace = AsyncVectorEnv.createBatchedObservationSpace(
            singleSpace: self.singleObservationSpace,
            numEnvs: numEnvs
        )

        self.actionSpace = AsyncVectorEnv.createBatchedActionSpace(
            singleSpace: self.singleActionSpace,
            numEnvs: numEnvs
        )
    }

    public init(
        envs: [any Env],
        copyObservations: Bool = true,
        autoresetMode: AutoresetMode = .nextStep
    ) throws {
        guard !envs.isEmpty else {
            throw GymnazoError.invalidNumEnvs(envs.count)
        }
        let probeEnv = envs[0]
        self.numEnvs = envs.count
        self.autoresetMode = autoresetMode
        self.copyObservations = copyObservations

        let actors = envs.enumerated().map { index, env in
            EnvironmentActor(index: index, envBox: UnsafeEnvBox(env: env), autoresetMode: autoresetMode)
        }
        self.actors = actors

        self.singleObservationSpace = probeEnv.observationSpace
        self.singleActionSpace = probeEnv.actionSpace
        self.renderMode = probeEnv.renderMode
        self.spec = probeEnv.spec
        self.observationShape = probeEnv.observationSpace.shape ?? [4]

        self.observationSpace = AsyncVectorEnv.createBatchedObservationSpace(
            singleSpace: self.singleObservationSpace,
            numEnvs: numEnvs
        )

        self.actionSpace = AsyncVectorEnv.createBatchedActionSpace(
            singleSpace: self.singleActionSpace,
            numEnvs: numEnvs
        )
    }

    nonisolated private static func blockingWait<T: Sendable>(
        _ work: @Sendable @escaping () async throws -> T
    ) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        let result = OSAllocatedUnfairLock<Result<T, Error>?>(initialState: nil)

        Task.detached {
            do {
                let value = try await work()
                result.withLock { $0 = .success(value) }
            } catch {
                result.withLock { $0 = .failure(error) }
            }
            semaphore.signal()
        }

        semaphore.wait()
        switch result.withLock({ $0! }) {
        case .success(let value):
            return value
        case .failure(let error):
            throw error
        }
    }

    /// Takes an action for each environment serially.
    ///
    ///
    /// - Parameter actions: Array of actions, one for each sub-environment.
    /// - Returns: Batched results containing observations, rewards, terminations, truncations, and infos.
    public func step(_ actions: [MLXArray]) throws -> VectorStepResult {
        guard !closed else {
            throw GymnazoError.vectorEnvClosed
        }
        guard actions.count == numEnvs else {
            throw GymnazoError.vectorEnvActionCountMismatch(
                expected: numEnvs,
                actual: actions.count
            )
        }
        let unsafeActions = actions.map(UnsafeAction.init)
        let actors = self.actors
        let results = try Self.blockingWait {
            try await Self.stepActorsParallel(actors: actors, actions: unsafeActions)
        }

        var observations: [[Float]] = Array(repeating: [], count: numEnvs)
        var rewards: [Float] = Array(repeating: 0.0, count: numEnvs)
        var terminations: [Bool] = Array(repeating: false, count: numEnvs)
        var truncations: [Bool] = Array(repeating: false, count: numEnvs)
        var infos: [Info] = Array(repeating: Info(), count: numEnvs)

        for result in results {
            let i = result.index
            observations[i] = result.observation
            rewards[i] = result.reward
            terminations[i] = result.terminated
            truncations[i] = result.truncated
            infos[i] = result.info
        }

        let batchedObs = batchObservations(observations)
        let batchedRewards = MLXArray(rewards)
        let batchedTerminations = MLXArray(terminations)
        let batchedTruncations = MLXArray(truncations)

        eval(batchedObs, batchedRewards, batchedTerminations, batchedTruncations)

        return VectorStepResult(
            observations: batchedObs,
            rewards: batchedRewards,
            terminations: batchedTerminations,
            truncations: batchedTruncations,
            infos: infos
        )
    }

    /// Asynchronously takes actions for each parallel environment using true parallelism.
    ///
    /// - Parameter actions: Array of actions, one for each sub-environment.
    /// - Returns: Batched results containing observations, rewards, terminations, truncations, and infos.
    public func stepAsync(_ actions: [MLXArray]) async throws -> VectorStepResult {
        guard !closed else {
            throw GymnazoError.vectorEnvClosed
        }
        guard actions.count == numEnvs else {
            throw GymnazoError.vectorEnvActionCountMismatch(
                expected: numEnvs,
                actual: actions.count
            )
        }
        let unsafeActions = actions.map(UnsafeAction.init)
        let results = try await Self.stepActorsParallel(actors: actors, actions: unsafeActions)

        var observations: [[Float]] = Array(repeating: [], count: numEnvs)
        var rewards: [Float] = Array(repeating: 0.0, count: numEnvs)
        var terminations: [Bool] = Array(repeating: false, count: numEnvs)
        var truncations: [Bool] = Array(repeating: false, count: numEnvs)
        var infos: [Info] = Array(repeating: Info(), count: numEnvs)

        for result in results {
            let i = result.index
            observations[i] = result.observation
            rewards[i] = result.reward
            terminations[i] = result.terminated
            truncations[i] = result.truncated
            infos[i] = result.info
        }

        let batchedObs = batchObservations(observations)
        let batchedRewards = MLXArray(rewards)
        let batchedTerminations = MLXArray(terminations)
        let batchedTruncations = MLXArray(truncations)

        eval(batchedObs, batchedRewards, batchedTerminations, batchedTruncations)

        return VectorStepResult(
            observations: batchedObs,
            rewards: batchedRewards,
            terminations: batchedTerminations,
            truncations: batchedTruncations,
            infos: infos
        )
    }

    nonisolated private static func stepActorsParallel(
        actors: [EnvironmentActor],
        actions: [UnsafeAction]
    ) async throws -> [EnvStepResult] {
        try await withThrowingTaskGroup(of: EnvStepResult.self) { group in
            for (i, actor) in actors.enumerated() {
                let action = actions[i]
                group.addTask {
                    try await actor.step(action)
                }
            }

            var results: [EnvStepResult] = []
            results.reserveCapacity(actors.count)
            for try await result in group {
                results.append(result)
            }
            return results.sorted { $0.index < $1.index }
        }
    }

    /// Resets all environments serially.
    ///
    /// For parallel execution, use `resetAsync(seed:options:)` instead.
    ///
    /// - Parameters:
    ///   - seed: Optional seed. If provided, seeds are `[seed, seed+1, ..., seed+n-1]`.
    ///   - options: Optional reset options dictionary.
    /// - Returns: Batched observations and info from all sub-environments.
    public func reset(seed: UInt64? = nil, options: EnvOptions? = nil) throws -> VectorResetResult {
        guard !closed else {
            throw GymnazoError.vectorEnvClosed
        }

        let actors = self.actors
        let results = try Self.blockingWait {
            try await Self.resetActorsParallel(actors: actors, seed: seed, options: options)
        }

        var observations: [[Float]] = Array(repeating: [], count: numEnvs)
        var infos: [Info] = Array(repeating: Info(), count: numEnvs)
        for result in results {
            observations[result.index] = result.observation
            infos[result.index] = result.info
        }

        let batchedObs = batchObservations(observations)
        eval(batchedObs)

        return VectorResetResult(observations: batchedObs, infos: infos)
    }

    /// Asynchronously resets all parallel environments.
    ///
    /// - Parameters:
    ///   - seed: Optional seed. If provided, seeds are `[seed, seed+1, ..., seed+n-1]`.
    ///   - options: Optional reset options dictionary.
    /// - Returns: Batched observations and info from all sub-environments.
    public func resetAsync(seed: UInt64? = nil, options: EnvOptions? = nil) async
        throws -> VectorResetResult
    {
        guard !closed else {
            throw GymnazoError.vectorEnvClosed
        }

        let results = try await Self.resetActorsParallel(actors: actors, seed: seed, options: options)

        var observations: [[Float]] = Array(repeating: [], count: numEnvs)
        var infos: [Info] = Array(repeating: Info(), count: numEnvs)
        for result in results {
            observations[result.index] = result.observation
            infos[result.index] = result.info
        }

        let batchedObs = batchObservations(observations)
        eval(batchedObs)

        return VectorResetResult(observations: batchedObs, infos: infos)
    }

    nonisolated private static func resetActorsParallel(
        actors: [EnvironmentActor],
        seed: UInt64?,
        options: EnvOptions?
    )
        async throws -> [EnvResetResult]
    {
        try await withThrowingTaskGroup(of: EnvResetResult.self) { group in
            for (i, actor) in actors.enumerated() {
                let envSeed: UInt64? = seed.map { $0 + UInt64(i) }
                group.addTask {
                    try await actor.reset(seed: envSeed, options: options)
                }
            }

            var results: [EnvResetResult] = []
            results.reserveCapacity(actors.count)
            for try await result in group {
                results.append(result)
            }
            return results.sorted { $0.index < $1.index }
        }
    }

    private func batchObservations(_ observations: [[Float]]) -> MLXArray {
        let flat = observations.flatMap { $0 }
        let batchedShape = [numEnvs] + observationShape
        return MLXArray(flat).reshaped(batchedShape)
    }

    nonisolated private static func closeActors(actors: [EnvironmentActor]) async {
        await withTaskGroup(of: Void.self) { group in
            for actor in actors {
                group.addTask {
                    await actor.close()
                }
            }
        }
    }

    public func close() {
        guard !closed else { return }

        let actors = self.actors
        do {
            try Self.blockingWait { await Self.closeActors(actors: actors) }
        } catch {
        }
        closed = true
    }

    private static func createBatchedObservationSpace(
        singleSpace: any Space,
        numEnvs: Int
    ) -> any Space {
        if let boxSpace = singleSpace as? Box {
            return batchedBox(space: boxSpace, numEnvs: numEnvs)
        }
        return singleSpace
    }

    private static func createBatchedActionSpace(
        singleSpace: any Space,
        numEnvs: Int
    ) -> any Space {
        if let discreteSpace = singleSpace as? Discrete {
            return MultiDiscrete(Array(repeating: discreteSpace.n, count: numEnvs))
        } else if let boxSpace = singleSpace as? Box {
            return batchedBox(space: boxSpace, numEnvs: numEnvs)
        }
        return singleSpace
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

