//
//  MountainCarContinuous.swift
//  Gymnazo
//

import Foundation
import MLX
#if canImport(SwiftUI)
import SwiftUI
#endif

/// The MountainCarContinuous environment from Gymnasium's classic control suite.
///
/// ## Description
///
/// The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
/// at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
/// that can be applied to the car in either direction. The goal of the MDP is to strategically
/// accelerate the car to reach the goal state on top of the right hill. There are two versions
/// of the mountain car domain in Gymnazo: one with discrete actions and one with continuous.
/// This version is the one with continuous actions.
///
/// This MDP first appeared in Andrew Moore's PhD Thesis (1990):
/// "Efficient Memory-based Learning for Robot Control", University of Cambridge.
///
/// ## Observation Space
///
/// The observation is an `MLXArray` with shape `(2,)` where the elements correspond to the following:
///
/// | Num | Observation                          | Min   | Max  | Unit          |
/// |-----|--------------------------------------|-------|------|---------------|
/// | 0   | position of the car along the x-axis | -1.2  | 0.6  | position (m)  |
/// | 1   | velocity of the car                  | -0.07 | 0.07 | velocity (v)  |
///
/// ## Action Space
///
/// The action is an `MLXArray` with shape `(1,)`, representing the directional force applied on the car.
/// The action is clipped in the range `[-1, 1]` and multiplied by a power of 0.0015.
///
/// ## Transition Dynamics
///
/// Given an action, the mountain car follows the following transition dynamics:
///
/// velocity(t+1) = velocity(t) + force * power - 0.0025 * cos(3 * position(t))
///
/// position(t+1) = position(t) + velocity(t+1)
///
/// where force is the action clipped to the range `[-1, 1]` and power is a constant 0.0015.
/// The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall.
/// The position is clipped to the range `[-1.2, 0.6]` and velocity is clipped to the range `[-0.07, 0.07]`.
///
/// ## Rewards
///
/// A negative reward of -0.1 * action² is received at each timestep to penalise for taking actions of
/// large magnitude. If the mountain car reaches the goal then a positive reward of +100 is added to
/// the negative reward for that timestep.
///
/// ## Starting State
///
/// The position of the car is assigned a uniform random value in `[-0.6, -0.4]`.
/// The starting velocity of the car is always assigned to 0.
///
/// ## Episode End
///
/// The episode ends if either of the following happens:
/// 1. **Termination:** The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
/// 2. **Truncation:** The length of the episode is 999.
///
/// ## Arguments
///
/// - `render_mode`: The render mode for visualization. Supported values: `"human"`, `"rgb_array"`, or `nil`.
/// - `goal_velocity`: The minimum velocity required at the goal position to terminate. Default is `0.0`.
///
/// ## Version History
///
/// - v0: Initial version release
public struct MountainCarContinuous: Env {
    public typealias Observation = MLXArray
    public typealias Action = MLXArray
    
    public let minPosition: Float = -1.2
    public let maxPosition: Float = 0.6
    public let maxSpeed: Float = 0.07
    public let goalPosition: Float = 0.45
    public let goalVelocity: Float = 0.0
    
    public let power: Float = 0.0015
    public let gravity: Float = 0.0025
    
    public var state: MLXArray? = nil
    
    public let action_space: Box
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
        
        // Continuous action space: force in [-1.0, 1.0]
        self.action_space = Box(
            low: MLXArray([-1.0] as [Float32]),
            high: MLXArray([1.0] as [Float32]),
            dtype: .float32
        )
        
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
    /// - Parameter action: An `MLXArray` with shape `(1,)` representing the force to apply (clipped to `[-1, 1]`).
    /// - Returns: A tuple containing the observation, reward, terminated flag, truncated flag, and info dictionary.
    public mutating func step(_ action: MLXArray) -> StepResult {
        guard let currentState = state else {
            fatalError("Call reset() before step()")
        }
        
        var position = currentState[0].item(Float.self)
        var velocity = currentState[1].item(Float.self)
        
        var force = action[0].item(Float.self)
        force = min(max(force, -1.0), 1.0)
        
        velocity += force * power + cos(3 * position) * (-gravity)
        velocity = min(max(velocity, -maxSpeed), maxSpeed)
        
        position += velocity
        position = min(max(position, minPosition), maxPosition)
        
        if position == minPosition && velocity < 0 {
            velocity = 0
        }
        
        self.state = MLXArray([position, velocity] as [Float32])
        
        let terminated = position >= goalPosition && velocity >= goalVelocity
        
        // Reward: 100 for reaching goal, minus action cost
        var reward: Double = 0.0
        if terminated {
            reward = 100.0
        }
        reward -= Double(force * force) * 0.1
        
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

