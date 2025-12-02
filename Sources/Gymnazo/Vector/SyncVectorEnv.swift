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
    public let num_envs: Int
    
    /// The sub-environments managed by this vector environment.
    private var envs: [any Env]
    
    /// Tracks which sub-environments need to be reset on the next step.
    private var needsReset: [Bool]
    
    /// Cached observations from the last step/reset for autoreset handling.
    private var lastObservations: [MLXArray]
    
    /// The observation space of a single sub-environment.
    public let single_observation_space: any Space
    
    /// The action space of a single sub-environment.
    public let single_action_space: any Space
    
    /// The batched observation space for all sub-environments.
    /// For Box spaces, this has shape `[num_envs, ...single_obs_shape]`.
    public private(set) var observation_space: any Space
    
    /// The batched action space for all sub-environments.
    /// For Discrete spaces, this becomes MultiDiscrete with `num_envs` dimensions.
    public private(set) var action_space: any Space
    
    /// The environment specification.
    public var spec: EnvSpec?
    
    /// The render mode for all sub-environments.
    public let render_mode: String?
    
    /// The autoreset mode used by this vector environment.
    public let autoreset_mode: AutoresetMode
    
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
        
        self.num_envs = envFns.count
        self.envs = envFns.map { $0() }
        self.needsReset = Array(repeating: true, count: envFns.count)
        self.copyObservations = copyObservations
        self.autoreset_mode = autoresetMode
        
        // Pre-allocate buffers
        self.rewardsBuffer = Array(repeating: 0.0, count: envFns.count)
        self.terminationsBuffer = Array(repeating: false, count: envFns.count)
        self.truncationsBuffer = Array(repeating: false, count: envFns.count)
        
        let firstEnv = self.envs[0]
        self.single_observation_space = firstEnv.observation_space
        self.single_action_space = firstEnv.action_space
        self.render_mode = firstEnv.render_mode
        self.spec = firstEnv.spec
        
        // Initialize lastObservations with placeholder
        self.lastObservations = Array(repeating: MLXArray([0.0] as [Float]), count: envFns.count)
        
        // Create batched observation space
        self.observation_space = SyncVectorEnv.createBatchedObservationSpace(
            singleSpace: self.single_observation_space,
            numEnvs: envFns.count
        )
        
        // Create batched action space
        self.action_space = SyncVectorEnv.createBatchedActionSpace(
            singleSpace: self.single_action_space,
            numEnvs: envFns.count
        )
        
        // Validate all environments have compatible spaces
        for (i, env) in self.envs.enumerated() {
            if let singleShape = single_observation_space.shape,
               let envShape = env.observation_space.shape {
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
            actions.count == num_envs,
            "Expected \(num_envs) actions, got \(actions.count)"
        )
        
        var observations: [MLXArray] = []
        observations.reserveCapacity(num_envs)
        
        var finalObservations: [Int: MLXArray] = [:]
        var finalInfos: [Int: [String: Any]] = [:]
        var infos: [String: Any] = [:]
        
        for i in 0..<num_envs {
            if needsReset[i] && autoreset_mode == .nextStep {
                let resetResult = envs[i].reset(seed: nil, options: nil)
                if let obs = resetResult.obs as? MLXArray {
                    lastObservations[i] = obs
                }
                needsReset[i] = false
            }
            
            let action = actions[i]
            let stepResult = stepEnvironment(index: i, action: action)
            
            let terminated = stepResult.terminated
            let truncated = stepResult.truncated
            let done = terminated || truncated
            
            guard let obs = stepResult.obs as? MLXArray else {
                fatalError("SyncVectorEnv currently only supports MLXArray observations")
            }
            
            if done && autoreset_mode == .nextStep {
                // Use identity operation to force a copy if needed
                finalObservations[i] = copyObservations ? (obs + MLXArray(Float(0))) : obs
                finalInfos[i] = stepResult.info
                needsReset[i] = true
            }
            
            observations.append(copyObservations ? (obs + MLXArray(Float(0))) : obs)
            lastObservations[i] = obs
            
            rewardsBuffer[i] = Float(stepResult.reward)
            terminationsBuffer[i] = terminated
            truncationsBuffer[i] = truncated
        }
        
        if !finalObservations.isEmpty {
            infos["final_observation"] = finalObservations
            infos["final_info"] = finalInfos
            infos["_final_observation_indices"] = Array(finalObservations.keys)
        }
        
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
            infos: infos
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
        observations.reserveCapacity(num_envs)
        let combinedInfo: [String: Any] = [:]
        
        for i in 0..<num_envs {
            let envSeed: UInt64? = seed.map { $0 + UInt64(i) }
            
            let resetResult = envs[i].reset(seed: envSeed, options: options)
            
            guard let obs = resetResult.obs as? MLXArray else {
                fatalError("SyncVectorEnv currently only supports MLXArray observations")
            }
            
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
    private func stepEnvironment(index: Int, action: Any) -> (obs: Any, reward: Double, terminated: Bool, truncated: Bool, info: [String: Any]) {
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
    private func stepWithAction<A>(index: Int, action: A) -> (obs: Any, reward: Double, terminated: Bool, truncated: Bool, info: [String: Any]) {
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
