//
//  VectorEnv.swift
//

import MLX

/// Autoreset mode for vector environments.
///
/// Determines when and how sub-environments are reset after termination or truncation.
public enum AutoresetMode: String, Sendable {
    /// Reset happens on the step after termination/truncation.
    /// The final observation is stored in the info dict under "final_observation".
    case nextStep = "next_step"
    
    /// Reset happens immediately in the same step.
    /// The observation returned is from the new episode.
    case sameStep = "same_step"
    
    /// No automatic reset. User must handle resets manually.
    case disabled = "disabled"
}

/// Result type for vectorized environment step operations.
public struct VectorStepResult {
    /// Batched observations from all sub-environments with shape `[num_envs, ...obs_shape]`.
    public let observations: MLXArray
    
    /// Rewards from all sub-environments with shape `[num_envs]`.
    public let rewards: MLXArray
    
    /// Termination flags for all sub-environments with shape `[num_envs]`.
    public let terminations: MLXArray
    
    /// Truncation flags for all sub-environments with shape `[num_envs]`.
    public let truncations: MLXArray
    
    /// Per-environment info dictionaries.
    public let infos: [Info]
    
    public init(
        observations: MLXArray,
        rewards: MLXArray,
        terminations: MLXArray,
        truncations: MLXArray,
        infos: [Info]
    ) {
        self.observations = observations
        self.rewards = rewards
        self.terminations = terminations
        self.truncations = truncations
        self.infos = infos
    }
}

/// Result type for vectorized environment reset operations.
public struct VectorResetResult {
    /// Batched observations from all sub-environments with shape `[num_envs, ...obs_shape]`.
    public let observations: MLXArray
    
    /// Per-environment info dictionaries.
    public let infos: [Info]
    
    public init(observations: MLXArray, infos: [Info]) {
        self.observations = observations
        self.infos = infos
    }
}

/// Protocol for vectorized environments that run multiple independent copies of
/// the same environment in parallel.
///
/// Vector environments provide a linear speed-up in steps taken per second by
/// sampling multiple sub-environments at the same time.
@MainActor
public protocol VectorEnv<Action>: AnyObject {
    associatedtype Action
    /// The number of sub-environments in the vector environment.
    var numEnvs: Int { get }
    
    /// The observation space of a single sub-environment.
    var singleObservationSpace: any Space<MLXArray> { get }
    
    /// The action space of a single sub-environment.
    var singleActionSpace: any Space<Action> { get }
    
    /// The batched observation space for all sub-environments.
    var observationSpace: any Space<MLXArray> { get }
    
    /// The batched action space for all sub-environments.
    var actionSpace: any Space<MLXArray> { get }
    
    /// The environment specification, if available.
    var spec: EnvSpec? { get set }
    
    /// The render mode for all sub-environments.
    var renderMode: RenderMode? { get }
    
    /// The autoreset mode used by this vector environment.
    var autoresetMode: AutoresetMode { get }
    
    /// Whether the vector environment has been closed.
    var closed: Bool { get }
    
    /// Take an action for each parallel environment.
    ///
    /// - Parameter actions: Array of actions, one for each sub-environment.
    /// - Returns: Batched results containing observations, rewards, terminations, truncations, and infos.
    func step(_ actions: [Action]) throws -> VectorStepResult
    
    /// Reset all parallel environments and return a batch of initial observations and info.
    ///
    /// - Parameters:
    ///   - seed: Optional seed for reproducibility. If an Int, seeds are `[seed, seed+1, ..., seed+n-1]`.
    ///   - options: Optional dictionary of reset options.
    /// - Returns: Batched observations and info from all sub-environments.
    func reset(seed: UInt64?, options: EnvOptions?) throws -> VectorResetResult
    
    /// Close all parallel environments and release resources.
    func close()
}

public extension VectorEnv {
    func reset(seed: UInt64? = nil) throws -> VectorResetResult {
        try reset(seed: seed, options: nil)
    }

    func reset() throws -> VectorResetResult {
        try reset(seed: nil, options: nil)
    }
}

