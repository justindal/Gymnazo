//
//  Pendulum.swift
//  Gymnazo
//

import Foundation
import MLX

#if canImport(SwiftUI)
import SwiftUI
#endif

/// A snapshot of the Pendulum state for serialization/debugging.
public struct PendulumSnapshot: Equatable, Sendable {
    public let theta: Float
    public let thetaDot: Float
    public let torque: Float?
    
    public init(theta: Float, thetaDot: Float, torque: Float? = nil) {
        self.theta = theta
        self.thetaDot = thetaDot
        self.torque = torque
    }
    
    public static var zero: PendulumSnapshot {
        PendulumSnapshot(theta: 0, thetaDot: 0, torque: nil)
    }
}

/// The Pendulum environment from Gymnasium's classic control suite.
///
/// ## Description
///
/// The inverted pendulum swingup problem is based on the classic problem in control theory.
/// The system consists of a pendulum attached at one end to a fixed point, and the other end
/// being free. The pendulum starts in a random position and the goal is to apply torque on
/// the free end to swing it into an upright position, with its center of gravity right above
/// the fixed point.
///
/// ## Observation Space
///
/// The observation is an `MLXArray` with shape `(3,)` where the elements correspond to:
///
/// | Num | Observation                  | Min  | Max  |
/// |-----|------------------------------|------|------|
/// | 0   | cos(θ)                       | -1.0 | 1.0  |
/// | 1   | sin(θ)                       | -1.0 | 1.0  |
/// | 2   | Angular velocity (θ̇)         | -8.0 | 8.0  |
///
/// ## Action Space
///
/// The action is an `MLXArray` with shape `(1,)` representing the torque applied to the pendulum.
/// The torque is clipped to the range `[-2.0, 2.0]`.
///
/// ## Rewards
///
/// The reward is calculated as: `-(θ² + 0.1 * θ̇² + 0.001 * τ²)`
/// where θ is normalized to `[-π, π]`, θ̇ is the angular velocity, and τ is the applied torque.
///
/// The reward is 0 when the pendulum is upright with zero velocity and zero torque.
/// The minimum reward is approximately -16.27.
///
/// ## Starting State
///
/// The starting state is a random angle θ in `[-π, π]` and random angular velocity θ̇ in `[-1, 1]`.
///
/// ## Episode End
///
/// The episode does not terminate naturally. It is truncated after 200 time steps
/// by the `TimeLimit` wrapper applied during registration.
///
/// ## Arguments
///
/// - `render_mode`: The render mode for visualization. Supported: `"human"`, `"rgb_array"`, or `nil`.
/// - `g`: Gravitational acceleration. Default is `10.0`.
public struct Pendulum: Env {
    public typealias Observation = MLXArray
    public typealias Action = MLXArray
    
    public let maxSpeed: Float = 8.0
    public let maxTorque: Float = 2.0
    public let dt: Float = 0.05
    public let g: Float
    public let m: Float = 1.0
    public let l: Float = 1.0
    
    public private(set) var state: (theta: Float, thetaDot: Float)? = nil
    
    public let action_space: Box
    public let observation_space: Box
    
    public var spec: EnvSpec? = nil
    public var render_mode: String? = nil
    
    private var _key: MLXArray?
    private var lastTorque: Float? = nil
    
    public static var metadata: [String: Any] {
        [
            "render_modes": ["human", "rgb_array"],
            "render_fps": 30,
        ]
    }
    
    public init(render_mode: String? = nil, g: Float = 10.0) {
        self.render_mode = render_mode
        self.g = g
        
        self.action_space = Box(
            low: MLXArray([-maxTorque] as [Float32]),
            high: MLXArray([maxTorque] as [Float32]),
            dtype: .float32
        )
        
        let high = MLXArray([1.0, 1.0, maxSpeed] as [Float32])
        self.observation_space = Box(
            low: -high,
            high: high,
            dtype: .float32
        )
    }
    
    public mutating func step(_ action: Action) -> Step<Observation> {
        guard let currentState = state else {
            fatalError("Call reset() before step()")
        }
        
        let theta = currentState.theta
        let thetaDot = currentState.thetaDot
        
        var torque = action[0].item(Float.self)
        torque = clip(torque, min: -maxTorque, max: maxTorque)
        lastTorque = torque
        
        let costs = angleNormalize(theta) * angleNormalize(theta)
            + 0.1 * thetaDot * thetaDot
            + 0.001 * torque * torque
        
        // Physics update: θ̈ = (3g / 2l) * sin(θ) + (3 / ml²) * τ
        let newThetaDot = thetaDot + (3 * g / (2 * l) * sin(theta) + 3.0 / (m * l * l) * torque) * dt
        let clippedThetaDot = clip(newThetaDot, min: -maxSpeed, max: maxSpeed)
        let newTheta = theta + clippedThetaDot * dt
        
        state = (theta: newTheta, thetaDot: clippedThetaDot)
        
        let observation = getObservation()
        let reward = Double(-costs)
        
        return Step(
            obs: observation,
            reward: reward,
            terminated: false,
            truncated: false,
            info: [:]
        )
    }
    
    public mutating func reset(seed: UInt64? = nil, options: [String: Any]? = nil) -> Reset<Observation> {
        if let seed = seed {
            _key = MLX.key(seed)
        } else if _key == nil {
            _key = MLX.key(UInt64.random(in: 0..<UInt64.max))
        }
        
        let xInit = (options?["x_init"] as? Float) ?? Float.pi
        let yInit = (options?["y_init"] as? Float) ?? 1.0
        
        let (k1, k2, nextKey) = splitKey3(_key!)
        _key = nextKey
        
        let theta = MLX.uniform(low: -xInit, high: xInit, [1], key: k1)[0].item(Float.self)
        let thetaDot = MLX.uniform(low: -yInit, high: yInit, [1], key: k2)[0].item(Float.self)
        
        state = (theta: theta, thetaDot: thetaDot)
        lastTorque = nil
        
        return Reset(obs: getObservation(), info: [:])
    }
    
    @discardableResult
    public func render() -> Any? {
        guard let mode = render_mode else { return nil }
        
        switch mode {
        case "human":
            #if canImport(SwiftUI)
            return PendulumView(snapshot: self.currentSnapshot)
            #else
            return nil
            #endif
        case "rgb_array":
            return nil
        default:
            return nil
        }
    }
    
    public var unwrapped: any Env { self }
    
    public var currentSnapshot: PendulumSnapshot? {
        guard let state = state else { return nil }
        return PendulumSnapshot(theta: state.theta, thetaDot: state.thetaDot, torque: lastTorque)
    }
    
    private func getObservation() -> MLXArray {
        guard let state = state else {
            fatalError("State is nil - call reset() first")
        }
        return MLXArray([cos(state.theta), sin(state.theta), state.thetaDot] as [Float32])
    }
    
    private func angleNormalize(_ x: Float) -> Float {
        let twoPi = 2 * Float.pi
        return ((x + Float.pi).truncatingRemainder(dividingBy: twoPi) + twoPi)
            .truncatingRemainder(dividingBy: twoPi) - Float.pi
    }
    
    private func clip(_ value: Float, min minVal: Float, max maxVal: Float) -> Float {
        Swift.min(Swift.max(value, minVal), maxVal)
    }
    
    private func splitKey3(_ key: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (k1, rest) = MLX.split(key: key)
        let (k2, k3) = MLX.split(key: rest)
        return (k1, k2, k3)
    }
}

#if canImport(SwiftUI)
public struct PendulumView: View {
    let snapshot: PendulumSnapshot?
    
    private let viewSize: CGFloat = 400
    private let pendulumLength: CGFloat = 150
    private let bobRadius: CGFloat = 20
    private let pivotRadius: CGFloat = 8
    
    public init(snapshot: PendulumSnapshot?) {
        self.snapshot = snapshot
    }
    
    public var body: some View {
        Canvas { context, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            
            // Draw background circle (range of motion)
            let rangePath = Path(ellipseIn: CGRect(
                x: center.x - pendulumLength,
                y: center.y - pendulumLength,
                width: pendulumLength * 2,
                height: pendulumLength * 2
            ))
            context.stroke(rangePath, with: .color(.gray.opacity(0.3)), lineWidth: 1)
            
            // Get theta from snapshot, default to hanging down (π)
            let theta = CGFloat(snapshot?.theta ?? Float.pi)
            
            // Calculate bob position
            // theta=0 is upright (pointing up), positive is counterclockwise
            let bobX = center.x + pendulumLength * sin(theta)
            let bobY = center.y - pendulumLength * cos(theta)
            let bobCenter = CGPoint(x: bobX, y: bobY)
            
            // Draw rod
            var rodPath = Path()
            rodPath.move(to: center)
            rodPath.addLine(to: bobCenter)
            context.stroke(rodPath, with: .color(.brown), style: StrokeStyle(lineWidth: 6, lineCap: .round))
            
            // Draw bob
            let bobRect = CGRect(
                x: bobCenter.x - bobRadius,
                y: bobCenter.y - bobRadius,
                width: bobRadius * 2,
                height: bobRadius * 2
            )
            context.fill(Path(ellipseIn: bobRect), with: .color(.red))
            context.stroke(Path(ellipseIn: bobRect), with: .color(.gray), lineWidth: 2)
            
            // Draw pivot
            let pivotRect = CGRect(
                x: center.x - pivotRadius,
                y: center.y - pivotRadius,
                width: pivotRadius * 2,
                height: pivotRadius * 2
            )
            context.fill(Path(ellipseIn: pivotRect), with: .color(.gray))
            
            // Draw torque indicator at bottom
            if let torque = snapshot?.torque {
                let indicatorY = size.height - 30
                let indicatorWidth = CGFloat(abs(torque) / 2.0) * 100
                let indicatorColor: Color = torque > 0 ? .green : .orange
                
                let indicatorRect = CGRect(
                    x: center.x - indicatorWidth / 2,
                    y: indicatorY - 5,
                    width: indicatorWidth,
                    height: 10
                )
                context.fill(Path(roundedRect: indicatorRect, cornerRadius: 3), with: .color(indicatorColor))
                
                // Arrow indicating direction
                var arrowPath = Path()
                if torque > 0 {
                    arrowPath.move(to: CGPoint(x: center.x + indicatorWidth / 2, y: indicatorY))
                    arrowPath.addLine(to: CGPoint(x: center.x + indicatorWidth / 2 + 10, y: indicatorY))
                } else {
                    arrowPath.move(to: CGPoint(x: center.x - indicatorWidth / 2, y: indicatorY))
                    arrowPath.addLine(to: CGPoint(x: center.x - indicatorWidth / 2 - 10, y: indicatorY))
                }
                context.stroke(arrowPath, with: .color(indicatorColor), lineWidth: 3)
            }
        }
        .frame(width: viewSize, height: viewSize)
        .background(Color.white)
    }
}

#Preview {
    VStack {
        PendulumView(snapshot: PendulumSnapshot(theta: 0, thetaDot: 0, torque: 1.0))
        PendulumView(snapshot: PendulumSnapshot(theta: Float.pi / 4, thetaDot: 0, torque: -0.5))
        PendulumView(snapshot: PendulumSnapshot(theta: Float.pi, thetaDot: 0, torque: nil))
    }
}
#endif
