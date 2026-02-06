import Foundation
import MLX

#if canImport(SwiftUI)
    import SwiftUI
#endif

/// Acrobot environment
///
/// Description:
/// The Acrobot environment is based on Sutton's work in "Generalization in Reinforcement Learning:
/// Successful Examples Using Sparse Coarse Coding" and Sutton and Barto's book.
/// The system consists of two links connected linearly to form a chain, with one end of
/// the chain fixed. The joint between the two links is actuated. The goal is to apply
/// torques on the actuated joint to swing the free end of the linear chain above a
/// given height while starting from the initial state of hanging downwards.
///
/// Observation Space:
/// The observation is a 6-element array:
/// - cos(theta1): [-1, 1]
/// - sin(theta1): [-1, 1]
/// - cos(theta2): [-1, 1]
/// - sin(theta2): [-1, 1]
/// - angular velocity of theta1: [-4π, 4π]
/// - angular velocity of theta2: [-9π, 9π]
///
/// where theta1 is the angle of the first joint (0 = pointing downward),
/// and theta2 is relative to the first link.
///
/// Action Space:
/// - 0: Apply -1 torque to the actuated joint
/// - 1: Apply 0 torque to the actuated joint
/// - 2: Apply +1 torque to the actuated joint
///
/// Rewards:
/// -1 for each step until termination, 0 on termination.
///
/// Starting State:
/// Each parameter (theta1, theta2, dtheta1, dtheta2) is initialized uniformly in [-0.1, 0.1].
///
/// Episode Termination:
/// The episode ends if:
/// - Termination: -cos(theta1) - cos(theta2 + theta1) > 1.0
/// - Truncation: Episode length > 500
public struct Acrobot: Env {
    public let dt: Float = 0.2
    public let linkLength1: Float = 1.0
    public let linkLength2: Float = 1.0
    public let linkMass1: Float = 1.0
    public let linkMass2: Float = 1.0
    public let linkCOMPos1: Float = 0.5
    public let linkCOMPos2: Float = 0.5
    public let linkMOI: Float = 1.0
    public let gravity: Float = 9.8

    public let maxVel1: Float = 4 * Float.pi
    public let maxVel2: Float = 9 * Float.pi

    public let availableTorques: [Float] = [-1.0, 0.0, 1.0]
    public let torqueNoiseMax: Float

    public var bookOrNips: String = "book"

    public var state: [Float]? = nil

    public let actionSpace: any Space
    public let observationSpace: any Space

    public var spec: EnvSpec? = nil
    public var renderMode: RenderMode? = nil

    private var _key: MLXArray?

    public static var metadata: [String: Any] {
        [
            "render_modes": ["human", "rgb_array"],
            "render_fps": 15,
        ]
    }

    public init(renderMode: RenderMode? = nil, torque_noise_max: Float = 0.0) {
        self.renderMode = renderMode
        self.torqueNoiseMax = torque_noise_max

        self.actionSpace = Discrete(n: 3)

        let high: [Float] = [1.0, 1.0, 1.0, 1.0, maxVel1, maxVel2]
        let low: [Float] = [-1.0, -1.0, -1.0, -1.0, -maxVel1, -maxVel2]

        self.observationSpace = Box(
            low: MLXArray(low),
            high: MLXArray(high),
            dtype: .float32
        )
    }
    
    private func toInt(_ action: MLXArray) -> Int {
        Int(action.item(Int32.self))
    }

    public mutating func step(_ action: MLXArray) throws -> Step {
        let a = toInt(action)
        guard let currentState = state else {
            throw GymnazoError.stepBeforeReset
        }

        guard a >= 0 && a < 3 else {
            throw GymnazoError.invalidAction("Invalid action: \(a). Must be 0, 1, or 2.")
        }

        var torque = availableTorques[a]

        if torqueNoiseMax > 0 {
            guard let key = _key else {
                throw GymnazoError.stepBeforeReset
            }
            let (noiseKey, nextKey) = MLX.split(key: key)
            _key = nextKey
            let noise = MLX.uniform(low: -torqueNoiseMax, high: torqueNoiseMax, [1], key: noiseKey)[
                0
            ].item(Float.self)
            torque += noise
        }

        let sAugmented = currentState + [torque]

        var ns = rk4(derivs: dsdt, y0: sAugmented, t: [0, dt])

        ns[0] = wrap(ns[0], min: -Float.pi, max: Float.pi)
        ns[1] = wrap(ns[1], min: -Float.pi, max: Float.pi)

        ns[2] = bound(ns[2], min: -maxVel1, max: maxVel1)
        ns[3] = bound(ns[3], min: -maxVel2, max: maxVel2)

        self.state = Array(ns[0..<4])

        let terminated = isTerminal()
        let reward: Double = terminated ? 0.0 : -1.0

        return Step(
            obs: try getObservation(),
            reward: reward,
            terminated: terminated,
            truncated: false,
            info: [:]
        )
    }

    public mutating func reset(seed: UInt64? = nil, options: EnvOptions? = nil) throws -> Reset {
        if let seed {
            self._key = MLX.key(seed)
        } else if self._key == nil {
            self._key = MLX.key(UInt64.random(in: 0...UInt64.max))
        }

        let (stepKey, nextKey) = MLX.split(key: self._key!)
        self._key = nextKey

        let low: Float = (options?["low"] as? Float) ?? -0.1
        let high: Float = (options?["high"] as? Float) ?? 0.1

        let randomState = MLX.uniform(low: low, high: high, [4], key: stepKey)
        eval(randomState)

        self.state = [
            randomState[0].item(Float.self),
            randomState[1].item(Float.self),
            randomState[2].item(Float.self),
            randomState[3].item(Float.self),
        ]

        return Reset(obs: try getObservation(), info: [:])
    }

    /// Get the 6D observation from the internal state
    private func getObservation() throws -> MLXArray {
        guard let s = state else {
            throw GymnazoError.invalidState("Call reset() before getting observation")
        }

        return MLXArray(
            [
                cos(s[0]),
                sin(s[0]),
                cos(s[1]),
                sin(s[1]),
                s[2],
                s[3],
            ] as [Float])
    }

    private func isTerminal() -> Bool {
        guard let s = state else { return false }
        return -cos(s[0]) - cos(s[1] + s[0]) > 1.0
    }

    /// Compute the derivatives for the dynamics
    /// s_augmented = [theta1, theta2, dtheta1, dtheta2, torque]
    /// TODO use MLX grad?
    private func dsdt(_ sAugmented: [Float]) -> [Float] {
        let m1 = linkMass1
        let m2 = linkMass2
        let l1 = linkLength1
        let lc1 = linkCOMPos1
        let lc2 = linkCOMPos2
        let I1 = linkMOI
        let I2 = linkMOI
        let g = gravity

        let a = sAugmented[4]
        let theta1 = sAugmented[0]
        let theta2 = sAugmented[1]
        let dtheta1 = sAugmented[2]
        let dtheta2 = sAugmented[3]

        let d1 = m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        let d2 = m2 * (lc2 * lc2 + l1 * lc2 * cos(theta2)) + I2
        let phi2 = m2 * lc2 * g * cos(theta1 + theta2 - Float.pi / 2.0)
        let phi1 =
            -m2 * l1 * lc2 * dtheta2 * dtheta2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - Float.pi / 2)
            + phi2

        let ddtheta2: Float
        if bookOrNips == "nips" {
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 * lc2 + I2 - d2 * d2 / d1)
        } else {
            ddtheta2 =
                (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 * dtheta1 * sin(theta2) - phi2)
                / (m2 * lc2 * lc2 + I2 - d2 * d2 / d1)
        }

        let ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        return [dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0]
    }

    /// 4th order Runge-Kutta integration
    private func rk4(derivs: ([Float]) -> [Float], y0: [Float], t: [Float]) -> [Float] {
        let dt = t[1] - t[0]
        let dt2 = dt / 2.0

        let k1 = derivs(y0)
        let y1 = zip(y0, k1).map { $0 + dt2 * $1 }

        let k2 = derivs(y1)
        let y2 = zip(y0, k2).map { $0 + dt2 * $1 }

        let k3 = derivs(y2)
        let y3 = zip(y0, k3).map { $0 + dt * $1 }

        let k4 = derivs(y3)

        var yout = [Float](repeating: 0, count: y0.count)
        for i in 0..<y0.count {
            yout[i] = y0[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        }

        return yout
    }

    /// Wrap angle x to be within [min, max]
    private func wrap(_ x: Float, min m: Float, max M: Float) -> Float {
        var result = x
        let diff = M - m
        while result > M {
            result -= diff
        }
        while result < m {
            result += diff
        }
        return result
    }

    /// Bound value x to be within [min, max]
    private func bound(_ x: Float, min m: Float, max M: Float) -> Float {
        return Swift.min(Swift.max(x, m), M)
    }

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

    public var currentSnapshot: AcrobotSnapshot {
        guard let s = state else { return AcrobotSnapshot.zero }
        return AcrobotSnapshot(
            theta1: s[0],
            theta2: s[1],
            linkLength1: linkLength1,
            linkLength2: linkLength2
        )
    }
}

public struct AcrobotSnapshot: Sendable, Equatable {
    public let theta1: Float
    public let theta2: Float
    public let linkLength1: Float
    public let linkLength2: Float

    public static let zero = AcrobotSnapshot(
        theta1: 0,
        theta2: 0,
        linkLength1: 1.0,
        linkLength2: 1.0
    )

    /// Returns the position of the first joint (end of link 1)
    public var p1: (x: Float, y: Float) {
        let x = linkLength1 * sin(theta1)
        let y = -linkLength1 * cos(theta1)
        return (x, y)
    }

    /// Returns the position of the free end (end of link 2)
    public var p2: (x: Float, y: Float) {
        let p1 = self.p1
        let x = p1.x + linkLength2 * sin(theta1 + theta2)
        let y = p1.y - linkLength2 * cos(theta1 + theta2)
        return (x, y)
    }

    /// Target height for termination
    public var targetHeight: Float {
        return linkLength1
    }
}

#if canImport(SwiftUI) && canImport(SpriteKit)
    import SpriteKit

    public struct AcrobotView: View {
        let snapshot: AcrobotSnapshot
        @State private var scene = AcrobotScene()

        public init(snapshot: AcrobotSnapshot) {
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
                SwiftUI.Text("Unsupported")
            }
        }
    }

    public class AcrobotScene: SKScene {
        var snapshot: AcrobotSnapshot = .zero

        private var link1: SKShapeNode?
        private var link2: SKShapeNode?
        private var joint0: SKShapeNode?
        private var joint1: SKShapeNode?
        private var joint2: SKShapeNode?
        private var targetLine: SKShapeNode?

        private let linkWidth: CGFloat = 10
        private let jointRadius: CGFloat = 8

        override init(size: CGSize = CGSize(width: 500, height: 500)) {
            super.init(size: size)
            self.scaleMode = .resizeFill
            self.backgroundColor = .white
        }

        required init?(coder aDecoder: NSCoder) {
            return nil
        }

        public func resize(to newSize: CGSize) {
            guard newSize.width > 0 && newSize.height > 0 else { return }
            self.size = newSize
            rebuildScene()
        }

        public func updateSnapshot(_ newSnapshot: AcrobotSnapshot) {
            self.snapshot = newSnapshot
            updatePositions()
        }

        override public func didMove(to view: SKView) {
            rebuildScene()
        }

        private func rebuildScene() {
            removeAllChildren()

            let bound = snapshot.linkLength1 + snapshot.linkLength2 + 0.2
            let scale = min(size.width, size.height) / CGFloat(bound * 2)
            let targetY = CGFloat(snapshot.targetHeight) * scale + size.height / 2

            let targetPath = CGMutablePath()
            targetPath.move(to: CGPoint(x: 0, y: targetY))
            targetPath.addLine(to: CGPoint(x: size.width, y: targetY))

            targetLine = SKShapeNode(path: targetPath)
            targetLine?.strokeColor = .black
            targetLine?.lineWidth = 2
            addChild(targetLine!)

            link1 = createLink()
            link2 = createLink()
            addChild(link1!)
            addChild(link2!)

            joint0 = SKShapeNode(circleOfRadius: jointRadius)
            joint0?.fillColor = SKColor(red: 0.8, green: 0.8, blue: 0.0, alpha: 1.0)
            joint0?.strokeColor = .black
            joint0?.lineWidth = 1
            addChild(joint0!)

            joint1 = SKShapeNode(circleOfRadius: jointRadius)
            joint1?.fillColor = SKColor(red: 0.0, green: 0.8, blue: 0.0, alpha: 1.0)
            joint1?.strokeColor = .black
            joint1?.lineWidth = 1
            addChild(joint1!)

            joint2 = SKShapeNode(circleOfRadius: jointRadius)
            joint2?.fillColor = SKColor(red: 0.8, green: 0.8, blue: 0.0, alpha: 1.0)
            joint2?.strokeColor = .black
            joint2?.lineWidth = 1
            addChild(joint2!)

            updatePositions()
        }

        private func createLink() -> SKShapeNode {
            let node = SKShapeNode()
            node.fillColor = SKColor(red: 0.0, green: 0.8, blue: 0.8, alpha: 1.0)
            node.strokeColor = .black
            node.lineWidth = 1
            return node
        }

        private func updatePositions() {
            let bound = snapshot.linkLength1 + snapshot.linkLength2 + 0.2
            let scale = min(size.width, size.height) / CGFloat(bound * 2)
            let offsetX = size.width / 2
            let offsetY = size.height / 2

            let origin = CGPoint(x: offsetX, y: offsetY)

            let p1 = snapshot.p1
            let p1Screen = CGPoint(
                x: offsetX + CGFloat(p1.x) * scale,
                y: offsetY + CGFloat(p1.y) * scale
            )

            let p2 = snapshot.p2
            let p2Screen = CGPoint(
                x: offsetX + CGFloat(p2.x) * scale,
                y: offsetY + CGFloat(p2.y) * scale
            )

            updateLinkShape(
                link1, from: origin, to: p1Screen, length: CGFloat(snapshot.linkLength1) * scale)

            updateLinkShape(
                link2, from: p1Screen, to: p2Screen, length: CGFloat(snapshot.linkLength2) * scale)

            joint0?.position = origin
            joint1?.position = p1Screen
            joint2?.position = p2Screen

            let targetY = CGFloat(snapshot.targetHeight) * scale + offsetY
            let targetPath = CGMutablePath()
            targetPath.move(to: CGPoint(x: 0, y: targetY))
            targetPath.addLine(to: CGPoint(x: size.width, y: targetY))
            targetLine?.path = targetPath
        }

        private func updateLinkShape(
            _ link: SKShapeNode?, from start: CGPoint, to end: CGPoint, length: CGFloat
        ) {
            guard let link = link else { return }

            let angle = atan2(end.y - start.y, end.x - start.x)
            let halfWidth = linkWidth / 2

            let path = CGMutablePath()
            path.addRect(CGRect(x: 0, y: -halfWidth, width: length, height: linkWidth))

            link.path = path
            link.position = start
            link.zRotation = angle
        }
    }
#endif
