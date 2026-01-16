//
//  SyncVectorEnv.swift
//

import MLX

/// Vectorized environment that serially runs multiple environments.
///
/// `SyncVectorEnv` manages multiple sub-environments and executes their `step` and `reset`
/// methods sequentially. It batches observations, rewards, terminations, and truncations
/// using MLX arrays for efficient processing.
///
/// ## Example
///
/// ```swift
/// // Create vector environment with 3 CartPole instances
/// let envs = SyncVectorEnv(envFns: [
///     { CartPole() },
///     { CartPole() },
///     { CartPole() }
/// ])
///
/// // Reset all environments
/// let (obs, _) = envs.reset(seed: 42)
/// // obs.shape == [3, 4] for 3 envs with 4-dimensional observations
///
/// // Step all environments
/// let result = envs.step([1, 0, 1])
/// // result.observations.shape == [3, 4]
/// // result.rewards.shape == [3]
/// ```
///
/// ## Autoreset Behavior
///
/// By default, `SyncVectorEnv` uses next-step autoreset. When a sub-environment
/// terminates or truncates:
/// 1. The final observation is stored in `infos["final_observation"]`
/// 2. The final info is stored in `infos["final_info"]`
/// 3. On the next step, the sub-environment is automatically reset
@MainActor
public final class SyncVectorEnv: VectorEnv {
    
    /// The number of sub-environments.
    public let numEnvs: Int
    
    /// The sub-environments managed by this vector environment.
    private var envs: [any Env]
    
    /// Tracks which sub-environments need to be reset on the next step.
    private var needsReset: [Bool]
    
    /// Cached observations from the last step/reset for autoreset handling.
    private var lastObservations: [MLXArray]
    
    /// The observation space of a single sub-environment.
    public let singleObservationSpace: any Space
    
    /// The action space of a single sub-environment.
    public let singleActionSpace: any Space
    
    /// The batched observation space for all sub-environments.
    /// For Box spaces, this has shape `[num_envs, ...single_obs_shape]`.
    public private(set) var observationSpace: any Space
    
    /// The batched action space for all sub-environments.
    /// For Discrete spaces, this becomes MultiDiscrete with `num_envs` dimensions.
    public private(set) var actionSpace: any Space
    
    /// The environment specification.
    public var spec: EnvSpec?
    
    /// The render mode for all sub-environments.
    public let renderMode: String?
    
    /// The autoreset mode used by this vector environment.
    public let autoresetMode: AutoresetMode
    
    /// Whether the vector environment has been closed.
    public private(set) var closed: Bool = false
    
    /// Whether to copy observations (prevents external mutation).
    private let copyObservations: Bool
    
    /// Pre-allocated array for collecting rewards during step.
    private var rewardsBuffer: [Float]
    
    /// Pre-allocated array for collecting terminations during step.
    private var terminationsBuffer: [Bool]
    
    /// Pre-allocated array for collecting truncations during step.
    private var truncationsBuffer: [Bool]
    
    /// Creates a new `SyncVectorEnv` from an array of environment factory functions.
    ///
    /// - Parameters:
    ///   - envFns: Array of closures that create environments.
    ///   - copyObservations: Whether to copy observations. Default is `true`.
    ///   - autoresetMode: The autoreset mode to use. Default is `.nextStep`.
    public init(
        envFns: [() -> any Env],
        copyObservations: Bool = true,
        autoresetMode: AutoresetMode = .nextStep
    ) {
        precondition(!envFns.isEmpty, "SyncVectorEnv requires at least one environment")
        
        self.numEnvs = envFns.count
        self.envs = envFns.map { $0() }
        self.needsReset = Array(repeating: true, count: envFns.count)
        self.copyObservations = copyObservations
        self.autoresetMode = autoresetMode
        
        // Pre-allocate buffers
        self.rewardsBuffer = Array(repeating: 0.0, count: envFns.count)
        self.terminationsBuffer = Array(repeating: false, count: envFns.count)
        self.truncationsBuffer = Array(repeating: false, count: envFns.count)
        
        let firstEnv = self.envs[0]
        self.singleObservationSpace = firstEnv.observationSpace
        self.singleActionSpace = firstEnv.actionSpace
        self.renderMode = firstEnv.renderMode
        self.spec = firstEnv.spec
        
        // Initialize lastObservations with placeholder
        self.lastObservations = Array(repeating: MLXArray([0.0] as [Float]), count: envFns.count)
        
        // Create batched observation space
        self.observationSpace = SyncVectorEnv.createBatchedObservationSpace(
            singleSpace: self.singleObservationSpace,
            numEnvs: envFns.count
        )
        
        // Create batched action space
        self.actionSpace = SyncVectorEnv.createBatchedActionSpace(
            singleSpace: self.singleActionSpace,
            numEnvs: envFns.count
        )
        
        // Validate all environments have compatible spaces
        for (i, env) in self.envs.enumerated() {
            if let singleShape = singleObservationSpace.shape,
               let envShape = env.observationSpace.shape {
                precondition(
                    singleShape == envShape,
                    "Environment \(i) has incompatible observation space shape: \(envShape) vs \(singleShape)"
                )
            }
        }
    }
    
    /// Creates a new `SyncVectorEnv` from pre-created environments.
    ///
    /// - Parameters:
    ///   - envs: Array of pre-created environments.
    ///   - copyObservations: Whether to copy observations. Default is `true`.
    ///   - autoresetMode: The autoreset mode to use. Default is `.nextStep`.
    public convenience init(
        envs: [any Env],
        copyObservations: Bool = true,
        autoresetMode: AutoresetMode = .nextStep
    ) {
        let envsCopy = envs
        var index = 0
        self.init(
            envFns: envs.map { _ in
                let currentIndex = index
                index += 1
                return { envsCopy[currentIndex] }
            },
            copyObservations: copyObservations,
            autoresetMode: autoresetMode
        )
        self.envs = envs
    }
    
    /// Takes an action for each parallel environment.
    ///
    /// - Parameter actions: Array of actions, one for each sub-environment.
    /// - Returns: Batched results containing observations, rewards, terminations, truncations, and infos.
    public func step(_ actions: [Any]) -> VectorStepResult {
        precondition(!closed, "Cannot step a closed vector environment")
        precondition(
            actions.count == numEnvs,
            "Expected \(numEnvs) actions, got \(actions.count)"
        )
        
        var observations: [MLXArray] = []
        observations.reserveCapacity(numEnvs)
        
        var finalObservations: [Int: MLXArray] = [:]
        var finalInfos: [Int: Info] = [:]
        let infos = Info()
        
        for i in 0..<numEnvs {
            if needsReset[i] {
                if autoresetMode == .nextStep {
                    let resetResult = resetEnvironment(index: i, seed: nil, options: nil)
                    lastObservations[i] = resetResult.obs
                needsReset[i] = false
                } else {
                    precondition(false, "Cannot step environment \(i) because it needs reset")
                }
            }
            
            let action = actions[i]
            let stepResult = stepEnvironment(index: i, action: action)
            
            let terminated = stepResult.terminated
            let truncated = stepResult.truncated
            let done = terminated || truncated
            
            guard let obs = stepResult.obs as? MLXArray else {
                fatalError("SyncVectorEnv currently only supports MLXArray observations")
            }
            
            if done {
                let finalObs = copyObservations ? (obs + MLXArray(Float(0))) : obs
                finalObservations[i] = finalObs
                finalInfos[i] = stepResult.info

                if autoresetMode == .sameStep {
                    let resetResult = resetEnvironment(index: i, seed: nil, options: nil)
                    needsReset[i] = false
                    let returnedObs = copyObservations ? (resetResult.obs + MLXArray(Float(0))) : resetResult.obs
                    observations.append(returnedObs)
                    lastObservations[i] = resetResult.obs
                } else {
                needsReset[i] = true
                    observations.append(finalObs)
                    lastObservations[i] = obs
                }
            } else {
            observations.append(copyObservations ? (obs + MLXArray(Float(0))) : obs)
            lastObservations[i] = obs
            }
            
            rewardsBuffer[i] = Float(stepResult.reward)
            terminationsBuffer[i] = terminated
            truncationsBuffer[i] = truncated
        }
        
        let finals: VectorFinals? = finalObservations.isEmpty
            ? nil
            : VectorFinals(
                observations: finalObservations,
                infos: finalInfos,
                indices: finalObservations.keys.sorted()
            )
        
        let batchedObs = MLX.stacked(observations, axis: 0)
        let batchedRewards = MLXArray(rewardsBuffer)
        let batchedTerminations = MLXArray(terminationsBuffer)
        let batchedTruncations = MLXArray(truncationsBuffer)
        
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
    
    /// Resets all parallel environments and returns batched initial observations.
    ///
    /// - Parameters:
    ///   - seed: Optional seed. If provided, seeds are `[seed, seed+1, ..., seed+n-1]`.
    ///   - options: Optional reset options dictionary.
    /// - Returns: Batched observations and info from all sub-environments.
    public func reset(seed: UInt64? = nil, options: [String: Any]? = nil) -> VectorResetResult {
        precondition(!closed, "Cannot reset a closed vector environment")
        
        var observations: [MLXArray] = []
        observations.reserveCapacity(numEnvs)
        let combinedInfo = Info()
        
        for i in 0..<numEnvs {
            let envSeed: UInt64? = seed.map { $0 + UInt64(i) }
            
            let resetResult = resetEnvironment(index: i, seed: envSeed, options: options)
            let obs = resetResult.obs
            
            observations.append(copyObservations ? (obs + MLXArray(Float(0))) : obs)
            lastObservations[i] = obs
            needsReset[i] = false
        }
        
        let batchedObs = MLX.stacked(observations, axis: 0)
        
        eval(batchedObs)
        
        return VectorResetResult(
            observations: batchedObs,
            infos: combinedInfo
        )
    }
    
    /// Closes all sub-environments and releases resources.
    public func close() {
        guard !closed else { return }
        
        for env in envs {
            env.close()
        }
        
        closed = true
    }
    
    /// Steps a single environment with the given action.
    private func stepEnvironment(index: Int, action: Any) -> (obs: Any, reward: Double, terminated: Bool, truncated: Bool, info: Info) {
        if let intAction = action as? Int {
            return stepWithAction(index: index, action: intAction)
        } else if let mlxAction = action as? MLXArray {
            return stepWithAction(index: index, action: mlxAction)
        } else if let floatAction = action as? Float {
            return stepWithAction(index: index, action: floatAction)
        } else if let arrayAction = action as? [Float] {
            return stepWithAction(index: index, action: MLXArray(arrayAction))
        } else {
            fatalError("Unsupported action type: \(type(of: action))")
        }
    }
    
    /// Type-safe step helper.
    private func stepWithAction<A>(index: Int, action: A) -> (obs: Any, reward: Double, terminated: Bool, truncated: Bool, info: Info) {
        let env = envs[index]
        
        if var discreteEnv = env as? any Env<MLXArray, Int>, let intAction = action as? Int {
            let result = discreteEnv.step(intAction)
            envs[index] = discreteEnv as any Env
            return (obs: result.obs, reward: result.reward, terminated: result.terminated, truncated: result.truncated, info: result.info)
        } else if var continuousEnv = env as? any Env<MLXArray, MLXArray>, let mlxAction = action as? MLXArray {
            let result = continuousEnv.step(mlxAction)
            envs[index] = continuousEnv as any Env
            return (obs: result.obs, reward: result.reward, terminated: result.terminated, truncated: result.truncated, info: result.info)
        }
        
        fatalError("Could not step environment with action type \(type(of: action))")
    }

    private func resetEnvironment(index: Int, seed: UInt64?, options: [String: Any]?) -> Reset<MLXArray> {
        let env = envs[index]

        if var discreteEnv = env as? any Env<MLXArray, Int> {
            let result = discreteEnv.reset(seed: seed, options: options)
            envs[index] = discreteEnv as any Env
            return result
        }

        if var continuousEnv = env as? any Env<MLXArray, MLXArray> {
            let result = continuousEnv.reset(seed: seed, options: options)
            envs[index] = continuousEnv as any Env
            return result
        }

        fatalError("SyncVectorEnv currently only supports MLXArray observations")
    }
    
    /// Access the underlying environments (read-only).
    public var environments: [any Env] {
        envs
    }
    
    /// Creates a batched observation space from a single observation space.
    private static func createBatchedObservationSpace(singleSpace: any Space, numEnvs: Int) -> any Space {
        if let boxSpace = singleSpace as? Box {
            return batchedBox(space: boxSpace, numEnvs: numEnvs)
        }
        // For other space types, return the single space (less optimal but functional)
        return singleSpace
    }
    
    /// Creates a batched action space from a single action space.
    private static func createBatchedActionSpace(singleSpace: any Space, numEnvs: Int) -> any Space {
        if let discreteSpace = singleSpace as? Discrete {
            return MultiDiscrete(Array(repeating: discreteSpace.n, count: numEnvs))
        } else if let boxSpace = singleSpace as? Box {
            return batchedBox(space: boxSpace, numEnvs: numEnvs)
        }
        return singleSpace
    }
}

/// Creates a batched Box space from a single Box space.
///
/// - Parameters:
///   - space: The single-environment Box space.
///   - numEnvs: Number of environments in the batch.
/// - Returns: A new Box space with shape `[numEnvs, ...originalShape]`.
public func batchedBox(space: Box, numEnvs: Int) -> Box {
    guard let shape = space.shape else {
        return space
    }
    
    let batchedShape = [numEnvs] + shape
    
    let tiledLow = MLX.repeated(space.low.expandedDimensions(axis: 0), count: numEnvs, axis: 0)
    let tiledHigh = MLX.repeated(space.high.expandedDimensions(axis: 0), count: numEnvs, axis: 0)
    
    return Box(
        low: tiledLow.reshaped(batchedShape),
        high: tiledHigh.reshaped(batchedShape),
        dtype: space.dtype ?? .float32
    )
}
