//
//  MountainCar.swift
//
//  A car is on a one-dimensional track, positioned between two "mountains".
//  The goal is to drive up the mountain on the right; however, the car's engine
//  is not strong enough to scale the mountain in a single pass. Therefore, the
//  only way to succeed is to drive back and forth to build up momentum.
//

import Foundation
import MLX
#if canImport(SwiftUI)
import SwiftUI
#endif

/// MountainCar-v0 environment
/// 
/// Description:
/// The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
/// at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
/// that can be applied to the car in either direction. The goal of the MDP is to strategically
/// accelerate the car to reach the goal state on top of the right hill.
///
/// Observation Space:
/// The observation is a 2-element array:
/// - position: [-1.2, 0.6]
/// - velocity: [-0.07, 0.07]
///
/// Action Space:
/// - 0: Accelerate to the left
/// - 1: Don't accelerate
/// - 2: Accelerate to the right
///
/// Rewards:
/// The goal is to reach the flag placed on top of the right hill as quickly as possible.
/// Reward of -1 for each time step until the goal is reached.
///
/// Starting State:
/// The position is randomly initialized in [-0.6, -0.4] and velocity is 0.
///
/// Episode Termination:
/// The episode ends if:
/// - Termination: position >= 0.5 (goal reached)
/// - Truncation: Episode length > 200 (default max steps)
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
            // Not implemented yet
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

#if canImport(SwiftUI) && canImport(SpriteKit)
import SpriteKit

public struct MountainCarView: View {
    let snapshot: MountainCarSnapshot
    @State private var scene = MountainCarScene()
    
    public init(snapshot: MountainCarSnapshot) {
        self.snapshot = snapshot
    }
    
    public var body: some View {
        if #available(iOS 17.0, macOS 14.0, *) {
            GeometryReader { geometry in
                SpriteView(scene: scene)
                    .onChange(of: snapshot) {
                        scene.updateSnapshot(snapshot)
                    }
                    .onChange(of: geometry.size) {
                        scene.resize(to: geometry.size)
                    }
                    .onAppear {
                        scene.resize(to: geometry.size)
                        scene.updateSnapshot(snapshot)
                    }
            }
        } else {
            Text("Unsupported")
        }
    }
}

public class MountainCarScene: SKScene {
    var snapshot: MountainCarSnapshot = .zero
    
    private var mountainNode: SKShapeNode?
    private var carBody: SKShapeNode?
    private var leftWheel: SKShapeNode?
    private var rightWheel: SKShapeNode?
    private var flagPole: SKShapeNode?
    private var flag: SKShapeNode?
    
    private let padding: CGFloat = 20
    private let carScale: CGFloat = 40
    private let flagHeight: CGFloat = 40
    
    override init(size: CGSize = CGSize(width: 600, height: 400)) {
        super.init(size: size)
        self.scaleMode = .resizeFill
        self.backgroundColor = .white
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    public func resize(to newSize: CGSize) {
        guard newSize.width > 0 && newSize.height > 0 else { return }
        self.size = newSize
        rebuildScene()
    }
    
    public func updateSnapshot(_ newSnapshot: MountainCarSnapshot) {
        self.snapshot = newSnapshot
        updatePositions()
    }
    
    override public func didMove(to view: SKView) {
        rebuildScene()
    }
    
    private func rebuildScene() {
        removeAllChildren()
        
        mountainNode = createMountainPath()
        if let mountain = mountainNode {
            addChild(mountain)
        }
        
        createFlag()
        createCar()
        updatePositions()
    }
    
    private func createMountainPath() -> SKShapeNode {
        let path = CGMutablePath()
        let segments = 100
        let range = snapshot.maxPosition - snapshot.minPosition
        
        let trackBottom: CGFloat = 20
        let trackHeight = size.height - trackBottom - 40
        
        path.move(to: CGPoint(x: 0, y: 0))
        
        for i in 0...segments {
            let t = Float(i) / Float(segments)
            let position = snapshot.minPosition + t * range
            let height = MountainCar.height(at: position)
            
            let x = CGFloat(t) * size.width
            let y = trackBottom + CGFloat(height) * trackHeight
            
            if i == 0 {
                path.addLine(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }
        
        path.addLine(to: CGPoint(x: size.width, y: 0))
        path.closeSubpath()
        
        let node = SKShapeNode(path: path)
        node.fillColor = SKColor(red: 0.82, green: 0.71, blue: 0.55, alpha: 1.0)
        node.strokeColor = SKColor(red: 0.6, green: 0.5, blue: 0.4, alpha: 1.0)
        node.lineWidth = 2
        
        return node
    }
    
    private func createFlag() {
        let polePath = CGMutablePath()
        polePath.move(to: CGPoint(x: 0, y: 0))
        polePath.addLine(to: CGPoint(x: 0, y: flagHeight))
        
        flagPole = SKShapeNode(path: polePath)
        flagPole?.strokeColor = .black
        flagPole?.lineWidth = 2
        addChild(flagPole!)
        
        let flagPath = CGMutablePath()
        flagPath.move(to: CGPoint(x: 0, y: flagHeight))
        flagPath.addLine(to: CGPoint(x: 20, y: flagHeight - 10))
        flagPath.addLine(to: CGPoint(x: 0, y: flagHeight - 20))
        flagPath.closeSubpath()
        
        flag = SKShapeNode(path: flagPath)
        flag?.fillColor = SKColor(red: 1.0, green: 0.84, blue: 0.0, alpha: 1.0)
        flag?.strokeColor = .black
        flag?.lineWidth = 1
        addChild(flag!)
    }
    
    private func createCar() {
        let carWidth: CGFloat = carScale
        let carHeight: CGFloat = carScale * 0.4
        
        let bodyPath = CGMutablePath()
        bodyPath.addRect(CGRect(x: -carWidth/2, y: 0, width: carWidth, height: carHeight))
        
        carBody = SKShapeNode(path: bodyPath)
        carBody?.fillColor = .black
        carBody?.strokeColor = .black
        carBody?.lineWidth = 1
        addChild(carBody!)
        
        let wheelRadius: CGFloat = carScale * 0.15
        
        leftWheel = SKShapeNode(circleOfRadius: wheelRadius)
        leftWheel?.fillColor = SKColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0)
        leftWheel?.strokeColor = .black
        leftWheel?.lineWidth = 1
        addChild(leftWheel!)
        
        rightWheel = SKShapeNode(circleOfRadius: wheelRadius)
        rightWheel?.fillColor = SKColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0)
        rightWheel?.strokeColor = .black
        rightWheel?.lineWidth = 1
        addChild(rightWheel!)
    }
    
    private func updatePositions() {
        let range = snapshot.maxPosition - snapshot.minPosition
        
        let trackBottom: CGFloat = 20
        let trackHeight = size.height - trackBottom - 40
        
        let normalizedPos = (snapshot.position - snapshot.minPosition) / range
        let carX = CGFloat(normalizedPos) * size.width
        let carY = trackBottom + CGFloat(snapshot.height) * trackHeight
        
        let delta: Float = 0.01
        let heightBefore = MountainCar.height(at: snapshot.position - delta)
        let heightAfter = MountainCar.height(at: snapshot.position + delta)
        let slope = (heightAfter - heightBefore) / (2 * delta)
        let visualSlope = slope * Float(trackHeight) / Float(size.width) * Float(range)
        let angle = atan(visualSlope)
        
        let carHeight = carScale * 0.4
        let wheelRadius = carScale * 0.15
        
        carBody?.position = CGPoint(x: carX, y: carY + wheelRadius + carHeight/2)
        carBody?.zRotation = CGFloat(angle)
        
        let wheelOffset = carScale * 0.35
        let cosAngle = cos(CGFloat(angle))
        let sinAngle = sin(CGFloat(angle))
        
        leftWheel?.position = CGPoint(
            x: carX - wheelOffset * cosAngle,
            y: carY + wheelRadius - wheelOffset * sinAngle
        )
        
        rightWheel?.position = CGPoint(
            x: carX + wheelOffset * cosAngle,
            y: carY + wheelRadius + wheelOffset * sinAngle
        )
        
        let goalNormalized = (snapshot.goalPosition - snapshot.minPosition) / range
        let goalX = CGFloat(goalNormalized) * size.width
        let goalY = trackBottom + CGFloat(snapshot.goalHeight) * trackHeight
        
        flagPole?.position = CGPoint(x: goalX, y: goalY)
        flag?.position = CGPoint(x: goalX, y: goalY)
    }
}
#endif

