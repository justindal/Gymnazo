//
//  CartPole.swift
//

import Foundation
import MLX
import SpriteKit
#if canImport(SwiftUI)
import SwiftUI
#endif

public struct CartPole: Env {
    public typealias Observation = MLXArray
    public typealias Action = Int
    
    public let gravity: Float = 9.8
    public let masscart: Float = 1.0
    public let masspole: Float = 0.1
    public let total_mass: Float
    public let length: Float = 0.5
    public let polemass_length: Float
    public let force_mag: Float = 10.0
    public let tau: Float = 0.02
    public let kinematics_integrator: String = "euler"
    
    // Angle at which to fail the episode
    public let theta_threshold_radians: Float = 12 * 2 * Float.pi / 360
    public let x_threshold: Float = 2.4
    
    public var state: MLXArray? = nil
    public var steps_beyond_terminated: Int? = nil
    
    public let action_space: Discrete
    public let observation_space: Box
    
    public var spec: EnvSpec? = nil
    public var render_mode: String? = nil
    
    private var _key: MLXArray?
    
    public init(render_mode: String? = nil) {
        self.render_mode = render_mode
        
        self.total_mass = masscart + masspole
        self.polemass_length = masspole * length
        
        // Action Space: 0 = push left, 1 = push right
        self.action_space = Discrete(n: 2)
        
        // Observation Space: [x, x_dot, theta, theta_dot]
        let high: [Float] = [
            x_threshold * 2,
            Float.greatestFiniteMagnitude,
            theta_threshold_radians * 2,
            Float.greatestFiniteMagnitude
        ]
        
        self.observation_space = Box(
            low: -MLXArray(high),
            high: MLXArray(high),
            dtype: .float32
        )
    }
    
    public mutating func step(_ action: Int) -> StepResult {
        guard let currentState = state else {
            fatalError("Call reset() before step()")
        }
        
        let x = currentState[0].item(Float.self)
        let x_dot = currentState[1].item(Float.self)
        let theta = currentState[2].item(Float.self)
        let theta_dot = currentState[3].item(Float.self)
        
        let force = action == 1 ? force_mag : -force_mag
        let costheta = cos(theta)
        let sintheta = sin(theta)
        
        // Physics
        let temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        let thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))
        let xacc = temp - polemass_length * thetaacc * costheta / total_mass
        
        var new_x = x
        var new_x_dot = x_dot
        var new_theta = theta
        var new_theta_dot = theta_dot
        
        if kinematics_integrator == "euler" {
            new_x = x + tau * x_dot
            new_x_dot = x_dot + tau * xacc
            new_theta = theta + tau * theta_dot
            new_theta_dot = theta_dot + tau * thetaacc
        } else {
            // Semi-implicit euler
            new_x_dot = x_dot + tau * xacc
            new_x = x + tau * new_x_dot
            new_theta_dot = theta_dot + tau * thetaacc
            new_theta = theta + tau * new_theta_dot
        }
        
        self.state = MLXArray([new_x, new_x_dot, new_theta, new_theta_dot])
        
        let terminated = Bool(
            new_x < -x_threshold ||
            new_x > x_threshold ||
            new_theta < -theta_threshold_radians ||
            new_theta > theta_threshold_radians
        )
        
        var reward: Double = 0.0
        if !terminated {
            reward = 1.0
        } else if steps_beyond_terminated == nil {
            // Pole just fell
            steps_beyond_terminated = 0
            reward = 1.0
        } else {
            if steps_beyond_terminated == 0 {
                print("You are calling 'step()' even though this environment has already returned terminated = True. You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior.")
            }
            steps_beyond_terminated! += 1
            reward = 0.0
        }
        
        return (
            obs: self.state!,
            reward: reward,
            terminated: terminated,
            truncated: false,
            info: [:]
        )
    }
    
    public mutating func reset(seed: UInt64? = nil, options: [String : Any]? = nil) -> ResetResult {
        if let seed {
            self._key = MLX.key(seed)
        } else if self._key == nil {
            self._key = MLX.key(UInt64.random(in: 0...UInt64.max))
        }
        
        // MLX split key
        let (stepKey, nextKey) = MLX.split(key: self._key!)
        self._key = nextKey
        
        // Random state between -0.05 and 0.05
        self.state = MLX.uniform(low: -0.05, high: 0.05, [4], key: stepKey)
        self.steps_beyond_terminated = nil
        
        return (obs: self.state!, info: [:])
    }
    
    public func render() -> Any? {
        guard let mode = render_mode else { return nil }
        
        switch mode {
        case "human":
            #if canImport(SwiftUI)
            return CartPoleView(snapshot: self.currentSnapshot)
            #else
            return nil
            #endif
        case "rgb_array":
            // Not implemented yet (requires offscreen rendering)
            return nil
        default:
            return nil
        }
    }
    
    
    public var currentSnapshot: CartPoleSnapshot {
        guard let s = state else { return CartPoleSnapshot.zero }
        let x = s[0].item(Float.self)
        let theta = s[2].item(Float.self)
        return CartPoleSnapshot(x: x, theta: theta, x_threshold: x_threshold)
    }
}

public struct CartPoleSnapshot: Sendable, Equatable {
    public let x: Float
    public let theta: Float
    public let x_threshold: Float
    
    public static let zero = CartPoleSnapshot(x: 0, theta: 0, x_threshold: 2.4)
}

#if canImport(SwiftUI) && canImport(SpriteKit)
public struct CartPoleView: View {
    let snapshot: CartPoleSnapshot
    @State private var scene = CartPoleScene()
    
    public init(snapshot: CartPoleSnapshot) {
        self.snapshot = snapshot
    }
    
    public var body: some View {
        if #available(iOS 17.0, *) {
            SpriteView(scene: scene)
                .frame(minWidth: 400, minHeight: 300)
                .onChange(of: snapshot) {
                    scene.updateSnapshot(snapshot)
                }
                .onAppear {
                    scene.updateSnapshot(snapshot)
                }
        } else {
            // to be implemented
        }
    }
}

class CartPoleScene: SKScene {
    var snapshot: CartPoleSnapshot = .zero
    
    private let cart = SKShapeNode(rectOf: CGSize(width: 50, height: 30))
    private let pole = SKShapeNode(rectOf: CGSize(width: 10, height: 100))
    private let axle = SKShapeNode(circleOfRadius: 5)
    private let track = SKShapeNode(rectOf: CGSize(width: 600, height: 2))
    
    override init(size: CGSize = CGSize(width: 600, height: 400)) {
        super.init(size: size)
        self.scaleMode = .aspectFit
        self.backgroundColor = .white
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    func updateSnapshot(_ newSnapshot: CartPoleSnapshot) {
        self.snapshot = newSnapshot
        updatePositions()
    }
    
    override func didMove(to view: SKView) {
        setupScene()
        updatePositions()
    }
    
    private func setupScene() {
        // Ground Track
        track.fillColor = .black
        track.strokeColor = .black
        track.position = CGPoint(x: size.width / 2, y: 100)
        addChild(track)
        
        // Cart
        cart.fillColor = .black
        cart.strokeColor = .black
        addChild(cart)
        
        // Pole
        pole.fillColor = .brown
        pole.strokeColor = .brown
        let polePath = CGMutablePath()
        polePath.addRect(CGRect(x: -5, y: 0, width: 10, height: 100))
        pole.path = polePath
        addChild(pole)
        
        // Axle
        axle.fillColor = .cyan
        axle.strokeColor = .cyan
        addChild(axle)
    }
    
    private func updatePositions() {
        let worldWidth = Float(snapshot.x_threshold * 2)
        let scale = Float(size.width) / worldWidth
        
        let cartX = Float(size.width) / 2 + snapshot.x * scale
        let cartY: Float = 100
        
        cart.position = CGPoint(x: Double(cartX), y: Double(cartY))
        pole.position = CGPoint(x: Double(cartX), y: Double(cartY))
        axle.position = CGPoint(x: Double(cartX), y: Double(cartY))
        
        pole.zRotation = CGFloat(-snapshot.theta)
    }
}
#endif
