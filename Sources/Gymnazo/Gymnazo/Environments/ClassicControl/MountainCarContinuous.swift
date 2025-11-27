//
//  MountainCarContinuous.swift
//
//  Continuous action version of the MountainCar environment.
//  The car can apply any force in the range [-1, 1].
//

import Foundation
import MLX
#if canImport(SwiftUI)
import SwiftUI
#endif

/// MountainCarContinuous-v0 environment
/// 
/// Description:
/// The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
/// at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
/// that can be applied to the car in either direction. Unlike the discrete version, this
/// environment accepts continuous actions.
///
/// Observation Space:
/// The observation is a 2-element array:
/// - position: [-1.2, 0.6]
/// - velocity: [-0.07, 0.07]
///
/// Action Space:
/// Continuous action in [-1.0, 1.0] representing the force applied to the car.
/// Negative values push left, positive values push right.
///
/// Rewards:
/// Reward of 100 for reaching the goal, minus the squared sum of actions from start to goal.
/// This reward function encourages reaching the goal with minimal energy expenditure.
///
/// Starting State:
/// The position is randomly initialized in [-0.6, -0.4] and velocity is 0.
///
/// Episode Termination:
/// The episode ends if:
/// - Termination: position >= 0.45 (goal reached)
/// - Truncation: Episode length > 999 (default max steps)
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

