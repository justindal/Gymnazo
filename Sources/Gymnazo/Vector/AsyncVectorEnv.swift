//
//  AsyncVectorEnv.swift
//

import Dispatch
import MLX

/// Sendable result type for step operations crossing actor boundaries.
public struct EnvStepResult: Sendable {
    public let index: Int
    public let observation: [Float]
    public let reward: Float
    public let terminated: Bool
    public let truncated: Bool
    
    public init(index: Int, observation: [Float], reward: Float, terminated: Bool, truncated: Bool) {
        self.index = index
        self.observation = observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
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

/// Wrapper to allow environments to cross actor boundaries.
/// This is safe because each environment is only accessed by a single actor.
public final class EnvBox: @unchecked Sendable {
    var env: any Env
    public init(_ env: any Env) { self.env = env }
}

/// Actor that wraps a single environment for isolated parallel execution.
public actor EnvironmentActor {
    private let envBox: EnvBox
    private var needsReset: Bool = true
    private let index: Int
    private let autoresetMode: AutoresetMode
    
    public init(index: Int, envBox: EnvBox, autoresetMode: AutoresetMode) {
        self.index = index
        self.envBox = envBox
        self.autoresetMode = autoresetMode
    }
    
    private var env: any Env {
        get { envBox.env }
        set { envBox.env = newValue }
    }
    
    public func step(_ action: Int) -> EnvStepResult {
        if needsReset && autoresetMode == .nextStep {
            _ = env.reset(seed: nil, options: nil)
            needsReset = false
        }
        
        if var discreteEnv = env as? any Env<MLXArray, Int> {
            let result = discreteEnv.step(action)
            env = discreteEnv as any Env
            
            let done = result.terminated || result.truncated
            if done && autoresetMode == .nextStep {
                needsReset = true
            }
            
            let obsArray: [Float] = result.obs.asArray(Float.self)
            return EnvStepResult(
                index: index,
                observation: obsArray,
                reward: Float(result.reward),
                terminated: result.terminated,
                truncated: result.truncated
            )
        }
        
        fatalError("EnvironmentActor only supports discrete action environments")
    }
    
    public func stepContinuous(_ action: [Float]) -> EnvStepResult {
        if needsReset && autoresetMode == .nextStep {
            _ = env.reset(seed: nil, options: nil)
            needsReset = false
        }
        
        if var continuousEnv = env as? any Env<MLXArray, MLXArray> {
            let mlxAction = MLXArray(action)
            let result = continuousEnv.step(mlxAction)
            env = continuousEnv as any Env
            
            let done = result.terminated || result.truncated
            if done && autoresetMode == .nextStep {
                needsReset = true
            }
            
            let obsArray: [Float] = result.obs.asArray(Float.self)
            return EnvStepResult(
                index: index,
                observation: obsArray,
                reward: Float(result.reward),
                terminated: result.terminated,
                truncated: result.truncated
            )
        }
        
        fatalError("EnvironmentActor only supports continuous action environments")
    }
    
    public func reset(seed: UInt64?) -> EnvResetResult {
        let result = env.reset(seed: seed, options: nil)
        needsReset = false
        
        guard let obs = result.obs as? MLXArray else {
            fatalError("EnvironmentActor only supports MLXArray observations")
        }
        
        let obsArray: [Float] = obs.asArray(Float.self)
        return EnvResetResult(index: index, observation: obsArray)
    }
    
    public func close() {
        env.close()
    }
    
    public var observationShape: [Int]? {
        env.observation_space.shape
    }
    
    public var actionSpace: any Space {
        env.action_space
    }
    
    public var observationSpace: any Space {
        env.observation_space
    }
    
    public var renderMode: String? {
        env.render_mode
    }
    
    public var envSpec: EnvSpec? {
        env.spec
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
    
    private var lastObservations: [[Float]]
    
    public let single_observation_space: any Space
    
    public let single_action_space: any Space
    
    public private(set) var observation_space: any Space
    
    public private(set) var action_space: any Space
    
    public var spec: EnvSpec?
    
    public let render_mode: String?
    
    public let autoreset_mode: AutoresetMode
    
    public private(set) var closed: Bool = false
    
    private let observationShape: [Int]
    
    /// Creates a new `AsyncVectorEnv` from an array of environment factory functions.
    ///
    /// - Parameters:
    ///   - envFns: Array of closures that create environments.
    ///   - copyObservations: Ignored, kept for API compatibility.
    ///   - autoresetMode: The autoreset mode to use. Default is `.nextStep`.
    public init(
        envFns: [() -> any Env],
        copyObservations: Bool = true,
        autoresetMode: AutoresetMode = .nextStep
    ) {
        precondition(!envFns.isEmpty, "AsyncVectorEnv requires at least one environment")
        
        let envs = envFns.map { $0() }
        let envBoxes = envs.map { EnvBox($0) }
        
        self.num_envs = envs.count
        self.autoreset_mode = autoresetMode
        
        self.actors = envBoxes.enumerated().map { index, envBox in
            EnvironmentActor(index: index, envBox: envBox, autoresetMode: autoresetMode)
        }
        
        let firstEnv = envs[0]
        self.single_observation_space = firstEnv.observation_space
        self.single_action_space = firstEnv.action_space
        self.render_mode = firstEnv.render_mode
        self.spec = firstEnv.spec
        self.observationShape = firstEnv.observation_space.shape ?? [4]
        
        let obsSize = self.observationShape.reduce(1, *)
        self.lastObservations = Array(repeating: Array(repeating: Float(0), count: obsSize), count: envs.count)
        
        self.observation_space = AsyncVectorEnv.createBatchedObservationSpace(
            singleSpace: self.single_observation_space,
            numEnvs: envs.count
        )
        
        self.action_space = AsyncVectorEnv.createBatchedActionSpace(
            singleSpace: self.single_action_space,
            numEnvs: envs.count
        )
    }
    
    
    /// Creates a new `AsyncVectorEnv` from pre-created environments.
    ///
    /// - Parameters:
    ///   - envs: Array of pre-created environments.
    ///   - copyObservations: Ignored, kept for API compatibility.
    ///   - autoresetMode: The autoreset mode to use. Default is `.nextStep`.
    public convenience init(
        envs: [any Env],
        copyObservations: Bool = true,
        autoresetMode: AutoresetMode = .nextStep
    ) {
        self.init(envFns: envs.map { env in { env } }, autoresetMode: autoresetMode)
    }
    
    /// Takes an action for each parallel environment (synchronous wrapper).
    ///
    /// - Parameter actions: Array of actions, one for each sub-environment.
    /// - Returns: Batched results containing observations, rewards, terminations, truncations, and infos.
    public func step(_ actions: [Any]) -> VectorStepResult {
        precondition(!closed, "Cannot step a closed vector environment")
        precondition(actions.count == num_envs, "Expected \(num_envs) actions, got \(actions.count)")
        
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
        
        var observations: [[Float]] = Array(repeating: [], count: num_envs)
        var rewards: [Float] = Array(repeating: 0.0, count: num_envs)
        var terminations: [Bool] = Array(repeating: false, count: num_envs)
        var truncations: [Bool] = Array(repeating: false, count: num_envs)
        var finalObservations: [Int: [Float]] = [:]
        var infos: [String: Any] = [:]
        
        let semaphore = DispatchSemaphore(value: 0)
        
        Task {
            let results = await stepActorsParallel(intActions: intActions, floatActions: floatActions)
            
            for result in results {
                let i = result.index
                observations[i] = result.observation
                rewards[i] = result.reward
                terminations[i] = result.terminated
                truncations[i] = result.truncated
                
                if result.terminated || result.truncated {
                    finalObservations[i] = result.observation
                }
                
                lastObservations[i] = result.observation
            }
            
            semaphore.signal()
        }
        
        semaphore.wait()
        
        if !finalObservations.isEmpty {
            let mlxFinalObs = finalObservations.mapValues { MLXArray($0).reshaped(observationShape) }
            infos["final_observation"] = mlxFinalObs
            infos["_final_observation_indices"] = Array(finalObservations.keys)
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
    public func stepAsync(_ actions: [Any]) async -> VectorStepResult {
        precondition(!closed, "Cannot step a closed vector environment")
        precondition(actions.count == num_envs, "Expected \(num_envs) actions, got \(actions.count)")
        
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
        
        let results = await stepActorsParallel(intActions: intActions, floatActions: floatActions)
        
        var observations: [[Float]] = Array(repeating: [], count: num_envs)
        var rewards: [Float] = Array(repeating: 0.0, count: num_envs)
        var terminations: [Bool] = Array(repeating: false, count: num_envs)
        var truncations: [Bool] = Array(repeating: false, count: num_envs)
        var finalObservations: [Int: [Float]] = [:]
        var infos: [String: Any] = [:]
        
        for result in results {
            let i = result.index
            observations[i] = result.observation
            rewards[i] = result.reward
            terminations[i] = result.terminated
            truncations[i] = result.truncated
            
            if result.terminated || result.truncated {
                finalObservations[i] = result.observation
            }
            
            lastObservations[i] = result.observation
        }
        
        if !finalObservations.isEmpty {
            let mlxFinalObs = finalObservations.mapValues { MLXArray($0).reshaped(observationShape) }
            infos["final_observation"] = mlxFinalObs
            infos["_final_observation_indices"] = Array(finalObservations.keys)
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
    
    private nonisolated func stepActorsParallel(intActions: [Int?], floatActions: [[Float]?]) async -> [EnvStepResult] {
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
    
    /// Resets all parallel environments (synchronous wrapper).
    ///
    /// - Parameters:
    ///   - seed: Optional seed. If provided, seeds are `[seed, seed+1, ..., seed+n-1]`.
    ///   - options: Optional reset options dictionary.
    /// - Returns: Batched observations and info from all sub-environments.
    public func reset(seed: UInt64? = nil, options: [String: Any]? = nil) -> VectorResetResult {
        precondition(!closed, "Cannot reset a closed vector environment")
        
        var observations: [[Float]] = Array(repeating: [], count: num_envs)
        let combinedInfo: [String: Any] = [:]
        
        let semaphore = DispatchSemaphore(value: 0)
        
        Task {
            let results = await resetActorsParallel(seed: seed)
            
            for result in results {
                observations[result.index] = result.observation
                lastObservations[result.index] = result.observation
            }
            
            semaphore.signal()
        }
        
        semaphore.wait()
        
        let batchedObs = batchObservations(observations)
        eval(batchedObs)
        
        return VectorResetResult(observations: batchedObs, infos: combinedInfo)
    }
    
    /// Asynchronously resets all parallel environments.
    ///
    /// - Parameters:
    ///   - seed: Optional seed. If provided, seeds are `[seed, seed+1, ..., seed+n-1]`.
    ///   - options: Optional reset options dictionary.
    /// - Returns: Batched observations and info from all sub-environments.
    public func resetAsync(seed: UInt64? = nil, options: [String: Any]? = nil) async -> VectorResetResult {
        precondition(!closed, "Cannot reset a closed vector environment")
        
        let results = await resetActorsParallel(seed: seed)
        
        var observations: [[Float]] = Array(repeating: [], count: num_envs)
        let combinedInfo: [String: Any] = [:]
        
        for result in results {
            observations[result.index] = result.observation
            lastObservations[result.index] = result.observation
        }
        
        let batchedObs = batchObservations(observations)
        eval(batchedObs)
        
        return VectorResetResult(observations: batchedObs, infos: combinedInfo)
    }
    
    private nonisolated func resetActorsParallel(seed: UInt64?) async -> [EnvResetResult] {
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
    
    public func close() {
        guard !closed else { return }
        
        Task {
            for actor in actors {
                await actor.close()
            }
        }
        
        closed = true
    }
    
    public var environments: [any Env] {
        fatalError("Direct environment access not available in AsyncVectorEnv. Use actor methods instead.")
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
