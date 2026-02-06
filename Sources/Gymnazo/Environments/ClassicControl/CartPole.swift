import Foundation
import MLX
import SpriteKit

#if canImport(SwiftUI)
    import SwiftUI
#endif

/// Classic cart-pole system implemented by Rich Sutton et al.
///
/// This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
/// ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
/// A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
/// The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
/// in the left and right direction on the cart.
///
/// ## Description
///
/// The goal is to balance the pole by applying forces in the left and right direction on the cart.
/// The velocity that is reduced or increased by the applied force is not fixed and depends on the angle
/// the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it.
///
/// ## Action Space
///
/// The action is an `Int` with values `{0, 1}` indicating the direction of the fixed force the cart is pushed with:
///
/// | Action | Direction            |
/// |--------|----------------------|
/// | 0      | Push cart to the left  |
/// | 1      | Push cart to the right |
///
/// ## Observation Space
///
/// The observation is an `MLXArray` with shape `(4,)` containing:
///
/// | Index | Observation           | Min                 | Max               |
/// |-------|-----------------------|---------------------|-------------------|
/// | 0     | Cart Position         | -4.8                | 4.8               |
/// | 1     | Cart Velocity         | -Inf                | Inf               |
/// | 2     | Pole Angle (radians)  | ~-0.418 rad (-24°)  | ~0.418 rad (24°)  |
/// | 3     | Pole Angular Velocity | -Inf                | Inf               |
///
/// > Note: While the ranges above denote the possible values for the observation space,
/// > the episode terminates if the cart leaves `(-2.4, 2.4)` or the pole angle exceeds `±12°`.
///
/// ## Rewards
///
/// A reward of **+1** is given for every step taken, including the termination step.
/// The reward threshold is 500 for v1.
///
/// ## Starting State
///
/// All observations are assigned a uniformly random value in `(-0.05, 0.05)`.
///
/// ## Episode End
///
/// The episode ends if any one of the following occurs:
///
/// **Termination:**
/// 1. Pole Angle is greater than ±12°
/// 2. Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
///
/// **Truncation:**
/// - Episode length is greater than 500 (when wrapped with ``TimeLimit``)
///
/// ## Arguments
///
/// - `render_mode`: The render mode (`"human"` or `"rgb_array"`).
///
/// ## Version History
///
/// - v1: Initial Swift port. `max_time_steps` is 500.
///
/// ## References
///
/// - [Cart-pole equations](https://coneural.org/florian/papers/05_cart_pole.pdf)
/// - [Original implementation](https://perma.cc/C9ZM-652R)
public struct CartPole: Env {
    public let gravity: Float = 9.8
    public let masscart: Float = 1.0
    public let masspole: Float = 0.1
    public let total_mass: Float
    public let length: Float = 0.5
    public let polemass_length: Float
    public let force_mag: Float = 10.0
    public let tau: Float = 0.02
    public let kinematics_integrator: String = "euler"

    public let theta_threshold_radians: Float = 12 * 2 * Float.pi / 360
    public let x_threshold: Float = 2.4

    public var state: MLXArray? = nil
    public var steps_beyond_terminated: Int? = nil

    public let actionSpace: any Space
    public let observationSpace: any Space

    public var spec: EnvSpec? = nil
    public var renderMode: RenderMode? = nil

    private var _key: MLXArray?

    public init(renderMode: RenderMode? = nil) {
        self.renderMode = renderMode

        self.total_mass = masscart + masspole
        self.polemass_length = masspole * length

        self.actionSpace = Discrete(n: 2)

        let high: [Float] = [
            x_threshold * 2,
            Float.greatestFiniteMagnitude,
            theta_threshold_radians * 2,
            Float.greatestFiniteMagnitude,
        ]

        self.observationSpace = Box(
            low: -MLXArray(high),
            high: MLXArray(high),
            dtype: .float32
        )
    }
    
    private func toInt(_ action: MLXArray) -> Int {
        Int(action.item(Int32.self))
    }

    /// Takes a step in the environment using the given action.
    ///
    /// Applies the action force to the cart and simulates one timestep of physics.
    ///
    /// - Parameter action: The action to take: `0` (push left) or `1` (push right).
    /// - Returns: A tuple containing:
    ///   - `obs`: The new observation `[x, x_dot, theta, theta_dot]`
    ///   - `reward`: `+1.0` for each step (including termination)
    ///   - `terminated`: `true` if cart or pole exceeded thresholds
    ///   - `truncated`: Always `false` (truncation handled by wrappers)
    ///   - `info`: Empty dictionary
    public mutating func step(_ action: MLXArray) throws -> Step {
        let a = toInt(action)
        guard let currentState = state else {
            throw GymnazoError.stepBeforeReset
        }

        let x = currentState[0].item(Float.self)
        let x_dot = currentState[1].item(Float.self)
        let theta = currentState[2].item(Float.self)
        let theta_dot = currentState[3].item(Float.self)

        let force = a == 1 ? force_mag : -force_mag
        let costheta = cos(theta)
        let sintheta = sin(theta)

        let temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        let thetaacc =
            (gravity * sintheta - costheta * temp)
            / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))
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
            new_x_dot = x_dot + tau * xacc
            new_x = x + tau * new_x_dot
            new_theta_dot = theta_dot + tau * thetaacc
            new_theta = theta + tau * new_theta_dot
        }

        self.state = MLXArray([new_x, new_x_dot, new_theta, new_theta_dot])

        let terminated = Bool(
            new_x < -x_threshold || new_x > x_threshold || new_theta < -theta_threshold_radians
                || new_theta > theta_threshold_radians
        )

        var reward: Double = 0.0
        if !terminated {
            reward = 1.0
        } else if steps_beyond_terminated == nil {
            steps_beyond_terminated = 0
            reward = 1.0
        } else {
            if steps_beyond_terminated == 0 {
                print(
                    "You are calling 'step()' even though this environment has already returned terminated = True. You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            }
            steps_beyond_terminated! += 1
            reward = 0.0
        }

        return Step(
            obs: self.state!,
            reward: reward,
            terminated: terminated,
            truncated: false,
            info: [:]
        )
    }

    /// Resets the environment to an initial state.
    ///
    /// Initializes all state values to a random value in `(-0.05, 0.05)`.
    ///
    /// - Parameters:
    ///   - seed: Optional random seed for reproducibility.
    ///   - options: Optional dictionary (unused in CartPole).
    /// - Returns: A tuple containing:
    ///   - `obs`: The initial observation `[x, x_dot, theta, theta_dot]`
    ///   - `info`: Empty dictionary
    public mutating func reset(seed: UInt64? = nil, options: EnvOptions? = nil) throws -> Reset {
        if let seed {
            self._key = MLX.key(seed)
        } else if self._key == nil {
            self._key = MLX.key(UInt64.random(in: 0...UInt64.max))
        }

        let (stepKey, nextKey) = MLX.split(key: self._key!)
        self._key = nextKey

        self.state = MLX.uniform(low: -0.05, high: 0.05, [4], key: stepKey)
        self.steps_beyond_terminated = nil

        return Reset(obs: self.state!, info: [:])
    }

    /// Renders the environment.
    ///
    /// - Returns: Depends on `render_mode`:
    ///   - `"human"`: Returns a `CartPoleSnapshot` for creating a `CartPoleView`
    ///   - `"rgb_array"`: Not yet implemented, returns `nil`
    ///   - `nil`: Returns `nil`
    public func render() throws -> RenderOutput? {
        guard let mode = renderMode else { return nil }

        switch mode {
        case .human:
            #if canImport(SwiftUI)
                return .other(self.currentSnapshot)
            #else
                return nil
            #endif
        case .rgbArray:
            return nil
        case .ansi, .statePixels:
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
            return nil
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
            track.fillColor = .black
            track.strokeColor = .black
            track.position = CGPoint(x: size.width / 2, y: 100)
            addChild(track)

            cart.fillColor = .black
            cart.strokeColor = .black
            addChild(cart)

            pole.fillColor = .brown
            pole.strokeColor = .brown
            let polePath = CGMutablePath()
            polePath.addRect(CGRect(x: -5, y: 0, width: 10, height: 100))
            pole.path = polePath
            addChild(pole)

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
