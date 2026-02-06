//
//  FrameSkip.swift
//  Gymnazo
//

import MLX

/// Repeats actions for a specified number of frames and accumulates rewards.
///
/// This wrapper reduces the effective decision frequency of the agent by repeating
/// each action for multiple environment steps. The rewards from all skipped frames
/// are accumulated and returned as a single sum.
///
/// ## Overview
///
/// Frame skipping is commonly used in video game environments to:
/// - Reduce computational cost by making fewer decisions
/// - Match human reaction time scales
/// - Simplify the learning problem by reducing temporal resolution
///
/// ## Example
///
/// ```swift
/// // Create environment with frame skip of 2
/// var env = FrameSkip(env: CarRacing(), skip: 2)
/// // Each step now executes the action twice, summing rewards
/// ```
public struct FrameSkip: Wrapper {
    public var env: any Env
    
    /// Number of frames to skip (repeat the action for).
    public let skip: Int
    
    /// Creates a frame skip wrapper.
    ///
    /// - Parameters:
    ///   - env: The environment to wrap
    ///   - skip: Number of times to repeat each action (must be >= 1)
    public init(env: any Env, skip: Int = 2) throws {
        guard skip >= 1 else {
            throw GymnazoError.invalidFrameSkip(skip)
        }
        self.env = env
        self.skip = skip
    }
    
    public init(env: any Env) throws {
        try self.init(env: env, skip: 2)
    }
    
    public mutating func step(_ action: MLXArray) throws -> Step {
        var totalReward: Double = 0
        var lastResult: Step!
        
        for _ in 0..<skip {
            lastResult = try env.step(action)
            totalReward += lastResult.reward
            
            if lastResult.terminated || lastResult.truncated {
                break
            }
        }
        
        return Step(
            obs: lastResult.obs,
            reward: totalReward,
            terminated: lastResult.terminated,
            truncated: lastResult.truncated,
            info: lastResult.info
        )
    }
}


