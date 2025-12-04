//
//  MountainCar.swift
//  Gymnazo
//

import Foundation
import MLX
#if canImport(SwiftUI)
import SwiftUI
#endif

/// The MountainCar environment from Gymnasium's classic control suite.
///
/// ## Description
///
/// The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
/// at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
/// that can be applied to the car in either direction. The goal of the MDP is to strategically
/// accelerate the car to reach the goal state on top of the right hill. There are two versions
/// of the mountain car domain in Gymnazo: one with discrete actions and one with continuous.
/// This version is the one with discrete actions.
///
/// This MDP first appeared in Andrew Moore's PhD Thesis (1990):
/// "Efficient Memory-based Learning for Robot Control", University of Cambridge.
///
/// ## Observation Space
///
/// The observation is an `MLXArray` with shape `(2,)` where the elements correspond to the following:
///
/// | Num | Observation                          | Min   | Max  | Unit         |
/// |-----|--------------------------------------|-------|------|--------------|
/// | 0   | position of the car along the x-axis | -1.2  | 0.6  | position (m) |
/// | 1   | velocity of the car                  | -0.07 | 0.07 | velocity (v) |
///
/// ## Action Space
///
/// There are 3 discrete deterministic actions:
///
/// - 0: Accelerate to the left
/// - 1: Don't accelerate
/// - 2: Accelerate to the right
///
/// ## Transition Dynamics
///
/// Given an action, the mountain car follows the following transition dynamics:
///
/// velocity(t+1) = velocity(t) + (action - 1) * force - cos(3 * position(t)) * gravity
///
/// position(t+1) = position(t) + velocity(t+1)
///
/// where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity
/// set to 0 upon collision with the wall. The position is clipped to the range `[-1.2, 0.6]` and velocity
/// is clipped to the range `[-0.07, 0.07]`.
///
/// ## Rewards
///
/// The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent
/// is penalised with a reward of -1 for each timestep.
///
/// ## Starting State
///
/// The position of the car is assigned a uniform random value in `[-0.6, -0.4]`.
/// The starting velocity of the car is always assigned to 0.
///
/// ## Episode End
///
/// The episode ends if either of the following happens:
/// 1. **Termination:** The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
/// 2. **Truncation:** The length of the episode is 200.
///
/// ## Arguments
///
/// - `render_mode`: The render mode for visualization. Supported values: `"human"`, `"rgb_array"`, or `nil`.
/// - `goal_velocity`: The minimum velocity required at the goal position to terminate. Default is `0.0`.
///
/// ## Version History
///
/// - v0: Initial version release
public struct MountainCar: Env {
    public typealias Observation = MLXArray
    public typealias Action = Int
    
    public let minPosition: Float = -1.2
    public let maxPosition: Float = 0.6
    public let maxSpeed: Float = 0.07
    public let goalPosition: Float = 0.5
    public let goalVelocity: Float = 0.0
    
    public let force: Float = 0.001
    public let gravity: Float = 0.0025
    
    // State: [position, velocity]
    public var state: MLXArray? = nil
    
    public let action_space: Discrete
    public let observation_space: Box
    
    public var spec: EnvSpec? = nil
    public var render_mode: String? = nil
    
    private var _key: MLXArray?
    
    public static var metadata: [String: Any] {
        [
            "render_modes": ["human", "rgb_array"],
            "render_fps": 30,
        ]
    }
    
    public init(render_mode: String? = nil, goal_velocity: Float = 0.0) {
        self.render_mode = render_mode
        
        // Action Space: 0 = push left, 1 = no push, 2 = push right
        self.action_space = Discrete(n: 3)
        
        // Observation Space: [position, velocity]
        let low = MLXArray([minPosition, -maxSpeed] as [Float32])
        let high = MLXArray([maxPosition, maxSpeed] as [Float32])
        
        self.observation_space = Box(
            low: low,
            high: high,
            dtype: .float32
        )
    }
    
    /// Execute one time step within the environment.
    ///
    /// - Parameter action: An action to take (0: left, 1: no push, 2: right).
    /// - Returns: A tuple containing the observation, reward, terminated flag, truncated flag, and info dictionary.
    public mutating func step(_ action: Int) -> StepResult {
        guard let currentState = state else {
            fatalError("Call reset() before step()")
        }
        
        precondition(action >= 0 && action < 3, "Invalid action: \(action). Must be 0, 1, or 2.")
        
        var position = currentState[0].item(Float.self)
        var velocity = currentState[1].item(Float.self)
        
        // velocity += (action - 1) * force + cos(3 * position) * (-gravity)
        velocity += Float(action - 1) * force + cos(3 * position) * (-gravity)
        velocity = min(max(velocity, -maxSpeed), maxSpeed)
        
        position += velocity
        position = min(max(position, minPosition), maxPosition)
        
        // If at left boundary and moving left, stop
        if position == minPosition && velocity < 0 {
            velocity = 0
        }
        
        self.state = MLXArray([position, velocity] as [Float32])
        
        let terminated = position >= goalPosition && velocity >= goalVelocity
        
        // Reward is -1 for each step
        let reward: Double = -1.0
        
        return (
            obs: self.state!,
            reward: reward,
            terminated: terminated,
            truncated: false,
            info: [:]
        )
    }
    
    /// Reset the environment to an initial state.
    ///
    /// - Parameters:
    ///   - seed: Optional random seed for reproducibility.
    ///   - options: Optional dictionary for custom reset bounds.
    /// - Returns: A tuple containing the initial observation and info dictionary.
    public mutating func reset(seed: UInt64? = nil, options: [String: Any]? = nil) -> ResetResult {
        if let seed {
            self._key = MLX.key(seed)
        } else if self._key == nil {
            self._key = MLX.key(UInt64.random(in: 0...UInt64.max))
        }
        
        let (stepKey, nextKey) = MLX.split(key: self._key!)
        self._key = nextKey
        
        // Random starting position in [-0.6, -0.4], velocity = 0
        let position = MLX.uniform(low: Float(-0.6), high: Float(-0.4), [1], key: stepKey)[0].item(Float.self)
        let velocity: Float = 0.0
        
        self.state = MLXArray([position, velocity] as [Float32])
        
        return (obs: self.state!, info: [:])
    }
    
    /// Render the environment.
    ///
    /// - Returns: A `MountainCarView` for human mode, `nil` otherwise.
    public func render() -> Any? {
        guard let mode = render_mode else { return nil }
        
        switch mode {
        case "human":
            #if canImport(SwiftUI)
            return MountainCarView(snapshot: self.currentSnapshot)
            #else
            return nil
            #endif
        case "rgb_array":
            return nil
        default:
            return nil
        }
    }
    
    public var currentSnapshot: MountainCarSnapshot {
        guard let s = state else { return MountainCarSnapshot.zero }
        let position = s[0].item(Float.self)
        let velocity = s[1].item(Float.self)
        return MountainCarSnapshot(
            position: position,
            velocity: velocity,
            minPosition: minPosition,
            maxPosition: maxPosition,
            goalPosition: goalPosition
        )
    }
    
    public static func height(at position: Float) -> Float {
        return sin(3 * position) * 0.45 + 0.55
    }
}

public struct MountainCarSnapshot: Sendable, Equatable {
    public let position: Float
    public let velocity: Float
    public let minPosition: Float
    public let maxPosition: Float
    public let goalPosition: Float
    
    public static let zero = MountainCarSnapshot(
        position: -0.5,
        velocity: 0,
        minPosition: -1.2,
        maxPosition: 0.6,
        goalPosition: 0.5
    )
    
    public var height: Float {
        MountainCar.height(at: position)
    }
    
    public var goalHeight: Float {
        MountainCar.height(at: goalPosition)
    }
}