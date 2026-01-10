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
    public let final: EnvFinal?
    
    public init(
        index: Int,
        observation: [Float],
        reward: Float,
        terminated: Bool,
        truncated: Bool,
        info: Info,
        final: EnvFinal? = nil
    ) {
        self.index = index
        self.observation = observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
        self.final = final
    }
}

/// Sendable result type for reset operations crossing actor boundaries.
public struct EnvResetResult: Sendable {
    public let index: Int
    public let observation: [Float]
    
    public init(index: Int, observation: [Float]) {
        self.index = index
        self.observation = observation
    }
}

public struct EnvFinal: Sendable {
    public let observation: [Float]
    public let info: Info

    public init(observation: [Float], info: Info) {
        self.observation = observation
        self.info = info
    }
}

public struct UnsafeEnvBox: @unchecked Sendable {
    public var env: any Env
    public init(env: any Env) { self.env = env }
}

/// Actor that wraps a single environment for isolated parallel execution.
public actor EnvironmentActor {
    private var needsReset: Bool = true
    private let index: Int
    private let autoresetMode: AutoresetMode
    private var env: any Env
    
    public init(index: Int, envFn: @Sendable () -> any Env, autoresetMode: AutoresetMode) {
        self.index = index
        self.autoresetMode = autoresetMode
        self.env = envFn()
    }

    public init(index: Int, envBox: UnsafeEnvBox, autoresetMode: AutoresetMode) {
        self.index = index
        self.autoresetMode = autoresetMode
        self.env = envBox.env
    }

    private func resetIfNeeded() {
        guard needsReset else { return }

        guard autoresetMode == .nextStep else {
            fatalError("Cannot step environment \(index) because it needs reset")
        }

        if var discreteEnv = env as? any Env<MLXArray, Int> {
            _ = discreteEnv.reset(seed: nil, options: nil)
            env = discreteEnv as any Env
            needsReset = false
            return
        }

        if var continuousEnv = env as? any Env<MLXArray, MLXArray> {
            _ = continuousEnv.reset(seed: nil, options: nil)
            env = continuousEnv as any Env
            needsReset = false
            return
        }

        fatalError("EnvironmentActor only supports MLXArray observation environments")
    }
    
    public func step(_ action: Int) -> EnvStepResult {
        resetIfNeeded()
        
        if var discreteEnv = env as? any Env<MLXArray, Int> {
            let result = discreteEnv.step(action)
            
            let done = result.terminated || result.truncated
            
            let finalObsArray: [Float] = result.obs.asArray(Float.self)
            let final: EnvFinal? = done ? EnvFinal(observation: finalObsArray, info: result.info) : nil

            if done && autoresetMode == .sameStep {
                let resetResult = discreteEnv.reset(seed: nil, options: nil)
                env = discreteEnv as any Env
                needsReset = false
                let obsArray: [Float] = resetResult.obs.asArray(Float.self)
                return EnvStepResult(
                    index: index,
                    observation: obsArray,
                    reward: Float(result.reward),
                    terminated: result.terminated,
                    truncated: result.truncated,
                    info: resetResult.info,
                    final: final
                )
            }

            env = discreteEnv as any Env
            needsReset = done
            return EnvStepResult(
                index: index,
                observation: finalObsArray,
                reward: Float(result.reward),
                terminated: result.terminated,
                truncated: result.truncated,
                info: result.info,
                final: final
            )
        }
        
        fatalError("EnvironmentActor only supports discrete action environments")
    }
    
    public func stepContinuous(_ action: [Float]) -> EnvStepResult {
        resetIfNeeded()
        
        if var continuousEnv = env as? any Env<MLXArray, MLXArray> {
            let mlxAction = MLXArray(action)
            let result = continuousEnv.step(mlxAction)
            
            let done = result.terminated || result.truncated
            
            let finalObsArray: [Float] = result.obs.asArray(Float.self)
            let final: EnvFinal? = done ? EnvFinal(observation: finalObsArray, info: result.info) : nil

            if done && autoresetMode == .sameStep {
                let resetResult = continuousEnv.reset(seed: nil, options: nil)
                env = continuousEnv as any Env
                needsReset = false
                let obsArray: [Float] = resetResult.obs.asArray(Float.self)
                return EnvStepResult(
                    index: index,
                    observation: obsArray,
                    reward: Float(result.reward),
                    terminated: result.terminated,
                    truncated: result.truncated,
                    info: resetResult.info,
                    final: final
                )
            }

            env = continuousEnv as any Env
            needsReset = done
            return EnvStepResult(
                index: index,
                observation: finalObsArray,
                reward: Float(result.reward),
                terminated: result.terminated,
                truncated: result.truncated,
                info: result.info,
                final: final
            )
        }
        
        fatalError("EnvironmentActor only supports continuous action environments")
    }
    
    public func reset(seed: UInt64?) -> EnvResetResult {
        if var discreteEnv = env as? any Env<MLXArray, Int> {
            let result = discreteEnv.reset(seed: seed, options: nil)
            env = discreteEnv as any Env
            needsReset = false
            let obsArray: [Float] = result.obs.asArray(Float.self)
            return EnvResetResult(index: index, observation: obsArray)
        }

        if var continuousEnv = env as? any Env<MLXArray, MLXArray> {
            let result = continuousEnv.reset(seed: seed, options: nil)
            env = continuousEnv as any Env
            needsReset = false
            let obsArray: [Float] = result.obs.asArray(Float.self)
            return EnvResetResult(index: index, observation: obsArray)
        }

        fatalError("EnvironmentActor only supports MLXArray observation environments")
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
/// let result = await envs.stepAsync([1, 0, 1, 0])
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
    
    public let num_envs: Int

    private let actors: [EnvironmentActor]
    
    public let single_observation_space: any Space
    
    public let single_action_space: any Space
    
    public private(set) var observation_space: any Space
    
    public private(set) var action_space: any Space
    
    public var spec: EnvSpec?
    
    public let render_mode: String?
    
    public let autoreset_mode: AutoresetMode
    
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
    ) {
        precondition(!envFns.isEmpty, "AsyncVectorEnv requires at least one environment")
        
        var probeEnv = envFns[0]()
        self.num_envs = envFns.count
        self.autoreset_mode = autoresetMode
        self.copyObservations = copyObservations

        let actors = envFns.enumerated().map { index, envFn in
            EnvironmentActor(index: index, envFn: envFn, autoresetMode: autoresetMode)
        }
        self.actors = actors

        self.single_observation_space = probeEnv.observation_space
        self.single_action_space = probeEnv.action_space
        self.render_mode = probeEnv.render_mode
        self.spec = probeEnv.spec
        self.observationShape = probeEnv.observation_space.shape ?? [4]
        probeEnv.close()
        
        self.observation_space = AsyncVectorEnv.createBatchedObservationSpace(
            singleSpace: self.single_observation_space,
            numEnvs: num_envs
        )
        
        self.action_space = AsyncVectorEnv.createBatchedActionSpace(
            singleSpace: self.single_action_space,
            numEnvs: num_envs
        )
    }

    public init(
        envs: [any Env],
        copyObservations: Bool = true,
        autoresetMode: AutoresetMode = .nextStep
    ) {
        precondition(!envs.isEmpty, "AsyncVectorEnv requires at least one environment")
        
        let probeEnv = envs[0]
        self.num_envs = envs.count
        self.autoreset_mode = autoresetMode
        self.copyObservations = copyObservations

        let actors = envs.enumerated().map { index, env in
            EnvironmentActor(index: index, envBox: UnsafeEnvBox(env: env), autoresetMode: autoresetMode)
        }
        self.actors = actors

        self.single_observation_space = probeEnv.observation_space
        self.single_action_space = probeEnv.action_space
        self.render_mode = probeEnv.render_mode
        self.spec = probeEnv.spec
        self.observationShape = probeEnv.observation_space.shape ?? [4]
        
        self.observation_space = AsyncVectorEnv.createBatchedObservationSpace(
            singleSpace: self.single_observation_space,
            numEnvs: num_envs
        )
        
        self.action_space = AsyncVectorEnv.createBatchedActionSpace(
            singleSpace: self.single_action_space,
            numEnvs: num_envs
        )
    }

    private static func splitActions(_ actions: [Any]) -> (intActions: [Int?], floatActions: [[Float]?]) {
        var intActions: [Int?] = Array(repeating: nil, count: actions.count)
        var floatActions: [[Float]?] = Array(repeating: nil, count: actions.count)

        for (i, action) in actions.enumerated() {
            if let intAction = action as? Int {
                intActions[i] = intAction
            } else if let floatArray = action as? [Float] {
                floatActions[i] = floatArray
            } else if let mlxAction = action as? MLXArray {
                floatActions[i] = mlxAction.asArray(Float.self)
            } else {
                fatalError("Unsupported action type: \(type(of: action))")
            }
        }

        return (intActions, floatActions)
    }

    nonisolated private static func blockingWait<T: Sendable>(_ work: @Sendable @escaping () async -> T) -> T {
        let semaphore = DispatchSemaphore(value: 0)
        let result = OSAllocatedUnfairLock<T?>(initialState: nil)
        
        Task.detached {
            let value = await work()
            result.withLock { $0 = value }
            semaphore.signal()
        }
        
        semaphore.wait()
        return result.withLock { $0! }
    }
    
    /// Takes an action for each environment serially.
    ///
    ///
    /// - Parameter actions: Array of actions, one for each sub-environment.
    /// - Returns: Batched results containing observations, rewards, terminations, truncations, and infos.
    public func step(_ actions: [Any]) -> VectorStepResult {
        precondition(!closed, "Cannot step a closed vector environment")
        precondition(actions.count == num_envs, "Expected \(num_envs) actions, got \(actions.count)")

        let (intActions, floatActions) = Self.splitActions(actions)
        let actors = self.actors
        let results = Self.blockingWait {
            await Self.stepActorsParallel(actors: actors, intActions: intActions, floatActions: floatActions)
        }

        var observations: [[Float]] = Array(repeating: [], count: num_envs)
        var rewards: [Float] = Array(repeating: 0.0, count: num_envs)
        var terminations: [Bool] = Array(repeating: false, count: num_envs)
        var truncations: [Bool] = Array(repeating: false, count: num_envs)
        var finalObservations: [Int: [Float]] = [:]
        var finalInfos: [Int: Info] = [:]
        let infos = Info()

        for result in results {
            let i = result.index
            observations[i] = result.observation
            rewards[i] = result.reward
            terminations[i] = result.terminated
            truncations[i] = result.truncated

            if let final = result.final {
                finalObservations[i] = final.observation
                finalInfos[i] = final.info
            }
        }

        let finals: VectorFinals? = finalObservations.isEmpty
            ? nil
            : {
                let mlxFinalObs = finalObservations.mapValues { MLXArray($0).reshaped(observationShape) }
                return VectorFinals(observations: mlxFinalObs, infos: finalInfos, indices: finalObservations.keys.sorted())
            }()

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
            infos: infos,
            finals: finals
        )
    }
    
    /// Asynchronously takes actions for each parallel environment using true parallelism.
    ///
    /// - Parameter actions: Array of actions, one for each sub-environment.
    /// - Returns: Batched results containing observations, rewards, terminations, truncations, and infos.
    public func stepAsync(_ actions: [Any]) async -> VectorStepResult {
        precondition(!closed, "Cannot step a closed vector environment")
        precondition(actions.count == num_envs, "Expected \(num_envs) actions, got \(actions.count)")

        let (intActions, floatActions) = Self.splitActions(actions)
        let results = await Self.stepActorsParallel(actors: actors, intActions: intActions, floatActions: floatActions)
        
        var observations: [[Float]] = Array(repeating: [], count: num_envs)
        var rewards: [Float] = Array(repeating: 0.0, count: num_envs)
        var terminations: [Bool] = Array(repeating: false, count: num_envs)
        var truncations: [Bool] = Array(repeating: false, count: num_envs)
        var finalObservations: [Int: [Float]] = [:]
        var finalInfos: [Int: Info] = [:]
        let infos = Info()
        
        for result in results {
            let i = result.index
            observations[i] = result.observation
            rewards[i] = result.reward
            terminations[i] = result.terminated
            truncations[i] = result.truncated
            
            if let final = result.final {
                finalObservations[i] = final.observation
                finalInfos[i] = final.info
            }
        }
        
        let finals: VectorFinals? = finalObservations.isEmpty
            ? nil
            : {
                let mlxFinalObs = finalObservations.mapValues { MLXArray($0).reshaped(observationShape) }
                return VectorFinals(observations: mlxFinalObs, infos: finalInfos, indices: finalObservations.keys.sorted())
            }()
        
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
            infos: infos,
            finals: finals
        )
    }
    
    nonisolated private static func stepActorsParallel(
        actors: [EnvironmentActor],
        intActions: [Int?],
        floatActions: [[Float]?]
    ) async -> [EnvStepResult] {
        await withTaskGroup(of: EnvStepResult.self) { group in
            for (i, actor) in actors.enumerated() {
                if let intAction = intActions[i] {
                    group.addTask {
                        await actor.step(intAction)
                    }
                } else if let floatArray = floatActions[i] {
                    group.addTask {
                        await actor.stepContinuous(floatArray)
                    }
                }
            }
            
            var results: [EnvStepResult] = []
            results.reserveCapacity(actors.count)
            for await result in group {
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
    public func reset(seed: UInt64? = nil, options: [String: Any]? = nil) -> VectorResetResult {
        precondition(!closed, "Cannot reset a closed vector environment")

        let actors = self.actors
        let results = Self.blockingWait { await Self.resetActorsParallel(actors: actors, seed: seed) }

        var observations: [[Float]] = Array(repeating: [], count: num_envs)
        for result in results {
            observations[result.index] = result.observation
        }

        let batchedObs = batchObservations(observations)
        eval(batchedObs)

        return VectorResetResult(observations: batchedObs, infos: Info())
    }
    
    /// Asynchronously resets all parallel environments.
    ///
    /// - Parameters:
    ///   - seed: Optional seed. If provided, seeds are `[seed, seed+1, ..., seed+n-1]`.
    ///   - options: Optional reset options dictionary.
    /// - Returns: Batched observations and info from all sub-environments.
    public func resetAsync(seed: UInt64? = nil, options: [String: Any]? = nil) async -> VectorResetResult {
        precondition(!closed, "Cannot reset a closed vector environment")
        
        let results = await Self.resetActorsParallel(actors: actors, seed: seed)

        var observations: [[Float]] = Array(repeating: [], count: num_envs)
        for result in results {
            observations[result.index] = result.observation
        }

        let batchedObs = batchObservations(observations)
        eval(batchedObs)

        return VectorResetResult(observations: batchedObs, infos: Info())
    }
    
    nonisolated private static func resetActorsParallel(actors: [EnvironmentActor], seed: UInt64?) async -> [EnvResetResult] {
        await withTaskGroup(of: EnvResetResult.self) { group in
            for (i, actor) in actors.enumerated() {
                let envSeed: UInt64? = seed.map { $0 + UInt64(i) }
                group.addTask {
                    await actor.reset(seed: envSeed)
                }
            }
            
            var results: [EnvResetResult] = []
            results.reserveCapacity(actors.count)
            for await result in group {
                results.append(result)
            }
            return results.sorted { $0.index < $1.index }
        }
    }
    
    private func batchObservations(_ observations: [[Float]]) -> MLXArray {
        let flat = observations.flatMap { $0 }
        let batchedShape = [num_envs] + observationShape
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
        Self.blockingWait { await Self.closeActors(actors: actors) }
        closed = true
    }
    
    private static func createBatchedObservationSpace(singleSpace: any Space, numEnvs: Int) -> any Space {
        if let boxSpace = singleSpace as? Box {
            return batchedBox(space: boxSpace, numEnvs: numEnvs)
        }
        return singleSpace
    }
    
    private static func createBatchedActionSpace(singleSpace: any Space, numEnvs: Int) -> any Space {
        if let discreteSpace = singleSpace as? Discrete {
            return MultiDiscrete(Array(repeating: discreteSpace.n, count: numEnvs))
        } else if let boxSpace = singleSpace as? Box {
            return batchedBox(space: boxSpace, numEnvs: numEnvs)
        }
        return singleSpace
    }
}
