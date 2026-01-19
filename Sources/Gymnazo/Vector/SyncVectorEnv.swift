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
public final class SyncVectorEnv<Action>: VectorEnv {
    
    /// The number of sub-environments.
    public let numEnvs: Int
    
    /// The sub-environments managed by this vector environment.
    private var envs: [AnyEnv<MLXArray, Action>]
    
    /// Tracks which sub-environments need to be reset on the next step.
    private var needsReset: [Bool]
    
    /// Cached observations from the last step/reset for autoreset handling.
    private var lastObservations: [MLXArray]
    
    /// The observation space of a single sub-environment.
    public let singleObservationSpace: any Space<MLXArray>
    
    /// The action space of a single sub-environment.
    public let singleActionSpace: any Space<Action>
    
    /// The batched observation space for all sub-environments.
    /// For Box spaces, this has shape `[num_envs, ...single_obs_shape]`.
    public private(set) var observationSpace: any Space<MLXArray>
    
    /// The batched action space for all sub-environments.
    /// For Discrete spaces, this becomes MultiDiscrete with `num_envs` dimensions.
    public private(set) var actionSpace: any Space<MLXArray>
    
    /// The environment specification.
    public var spec: EnvSpec?
    
    /// The render mode for all sub-environments.
    public let renderMode: RenderMode?
    
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
    ) throws {
        guard !envFns.isEmpty else {
            throw GymnazoError.invalidNumEnvs(envFns.count)
        }
        
        self.numEnvs = envFns.count
        self.envs = try envFns.map { try Self.wrapEnv($0()) }
        self.needsReset = Array(repeating: true, count: envFns.count)
        self.copyObservations = copyObservations
        self.autoresetMode = autoresetMode
        
        self.rewardsBuffer = Array(repeating: 0.0, count: envFns.count)
        self.terminationsBuffer = Array(repeating: false, count: envFns.count)
        self.truncationsBuffer = Array(repeating: false, count: envFns.count)
        
        let firstEnv = self.envs[0]
        self.singleObservationSpace = firstEnv.observationSpace
        self.singleActionSpace = firstEnv.actionSpace
        self.renderMode = firstEnv.renderMode
        self.spec = firstEnv.spec
        
        self.lastObservations = Array(repeating: MLXArray([0.0] as [Float]), count: envFns.count)
        
        self.observationSpace = SyncVectorEnv.createBatchedObservationSpace(
            singleSpace: self.singleObservationSpace,
            numEnvs: envFns.count
        )
        
        self.actionSpace = SyncVectorEnv.createBatchedActionSpace(
            singleSpace: self.singleActionSpace,
            numEnvs: envFns.count
        )
        
        for (i, env) in self.envs.enumerated() {
            if let singleShape = singleObservationSpace.shape,
               let envShape = env.observationSpace.shape {
                guard singleShape == envShape else {
                    throw GymnazoError.vectorEnvIncompatibleObservationShape(
                        index: i,
                        expected: singleShape,
                        actual: envShape
                    )
                }
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
    ) throws {
        let envsCopy = envs
        var index = 0
        try self.init(
            envFns: envs.map { _ in
                let currentIndex = index
                index += 1
                return { envsCopy[currentIndex] }
            },
            copyObservations: copyObservations,
            autoresetMode: autoresetMode
        )
    }
    
    /// Takes an action for each parallel environment.
    ///
    /// - Parameter actions: Array of actions, one for each sub-environment.
    /// - Returns: Batched results containing observations, rewards, terminations, truncations, and infos.
    public func step(_ actions: [Action]) throws -> VectorStepResult {
        guard !closed else {
            throw GymnazoError.vectorEnvClosed
        }
        guard actions.count == numEnvs else {
            throw GymnazoError.vectorEnvActionCountMismatch(
                expected: numEnvs,
                actual: actions.count
            )
        }
        
        var observations: [MLXArray] = []
        observations.reserveCapacity(numEnvs)

        var infos = Array(repeating: Info(), count: numEnvs)
        
        for i in 0..<numEnvs {
            if needsReset[i] {
                if autoresetMode == .nextStep {
                    let resetResult = try envs[i].reset(seed: nil, options: nil)
                    lastObservations[i] = resetResult.obs
                    needsReset[i] = false
                } else {
                    throw GymnazoError.vectorEnvNeedsReset(index: i)
                }
            }

            let stepResult = try envs[i].step(actions[i])
            
            let terminated = stepResult.terminated
            let truncated = stepResult.truncated
            let done = terminated || truncated
            let obs = stepResult.obs
            
            if done {
                let finalObs = copyObservations ? (obs + MLXArray(Float(0))) : obs
                if autoresetMode == .sameStep {
                    let resetResult = try envs[i].reset(seed: nil, options: nil)
                    needsReset[i] = false
                    let returnedObs = copyObservations ? (resetResult.obs + MLXArray(Float(0))) : resetResult.obs
                    observations.append(returnedObs)
                    lastObservations[i] = resetResult.obs
                    var info = resetResult.info
                    if let value = sendableValue(finalObs) {
                        info["final_observation"] = value
                    }
                    info["final_info"] = .object(stepResult.info.storage)
                    infos[i] = info
                } else {
                    needsReset[i] = true
                    observations.append(finalObs)
                    lastObservations[i] = obs
                    var info = stepResult.info
                    if let value = sendableValue(finalObs) {
                        info["final_observation"] = value
                    }
                    info["final_info"] = .object(stepResult.info.storage)
                    infos[i] = info
                }
            } else {
                observations.append(copyObservations ? (obs + MLXArray(Float(0))) : obs)
                lastObservations[i] = obs
                infos[i] = stepResult.info
            }
            
            rewardsBuffer[i] = Float(stepResult.reward)
            terminationsBuffer[i] = terminated
            truncationsBuffer[i] = truncated
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
    public func reset(seed: UInt64? = nil, options: EnvOptions? = nil) throws -> VectorResetResult {
        guard !closed else {
            throw GymnazoError.vectorEnvClosed
        }
        
        var observations: [MLXArray] = []
        observations.reserveCapacity(numEnvs)
        var infos = Array(repeating: Info(), count: numEnvs)
        
        for i in 0..<numEnvs {
            let envSeed: UInt64? = seed.map { $0 + UInt64(i) }
            
            let resetResult = try envs[i].reset(seed: envSeed, options: options)
            let obs = resetResult.obs
            
            observations.append(copyObservations ? (obs + MLXArray(Float(0))) : obs)
            lastObservations[i] = obs
            needsReset[i] = false
            infos[i] = resetResult.info
        }
        
        let batchedObs = MLX.stacked(observations, axis: 0)
        
        eval(batchedObs)
        
        return VectorResetResult(
            observations: batchedObs,
            infos: infos
        )
    }
    
    /// Closes all sub-environments and releases resources.
    public func close() {
        guard !closed else { return }
        
        for index in envs.indices {
            envs[index].close()
        }
        
        closed = true
    }
    
    private static func wrapEnv(_ env: any Env) throws -> AnyEnv<MLXArray, Action> {
        guard let typed = env as? any Env<MLXArray, Action> else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "Env<MLXArray, \(Action.self)>",
                actual: String(describing: type(of: env))
            )
        }
        return AnyEnv(typed)
    }

    /// Creates a batched observation space from a single observation space.
    private static func createBatchedObservationSpace(
        singleSpace: any Space<MLXArray>,
        numEnvs: Int
    ) -> any Space<MLXArray> {
        if let boxSpace = singleSpace as? Box {
            return batchedBox(space: boxSpace, numEnvs: numEnvs)
        }
        return singleSpace
    }
    
    /// Creates a batched action space from a single action space.
    private static func createBatchedActionSpace(
        singleSpace: any Space<Action>,
        numEnvs: Int
    ) -> any Space<MLXArray> {
        if let discreteSpace = singleSpace as? Discrete {
            return MultiDiscrete(Array(repeating: discreteSpace.n, count: numEnvs))
        } else if let boxSpace = singleSpace as? Box {
            return batchedBox(space: boxSpace, numEnvs: numEnvs)
        }
        if let space = singleSpace as? any Space<MLXArray> {
            return space
        }
        fatalError("Unsupported action space for vectorization")
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
