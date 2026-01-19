#if canImport(SwiftUI) && canImport(SpriteKit)
import SwiftUI
import SpriteKit

public struct LunarLanderView: View {
    let snapshot: LunarLanderSnapshot
    @State private var scene = LunarLanderScene()
    
    public init(snapshot: LunarLanderSnapshot) {
        self.snapshot = snapshot
    }
    
    public var body: some View {
        if #available(iOS 17.0, macOS 14.0, tvOS 17.0, *) {
            GeometryReader { geometry in
                SpriteView(scene: scene)
                    .onChange(of: snapshot) {
                        scene.updateSnapshot(snapshot)
                    }
                    .onAppear {
                        scene.size = geometry.size
                        scene.updateSnapshot(snapshot)
                    }
            }
            .aspectRatio(600 / 400, contentMode: .fit)
        } else {
            SwiftUI.Text("LunarLander rendering requires iOS 17+ / macOS 14+")
                .foregroundColor(.secondary)
        }
    }
}

public class LunarLanderScene: SKScene {
    private var snapshot: LunarLanderSnapshot = .zero
    
    private var terrainNode: SKShapeNode?
    private var landerNode: SKShapeNode?
    private var leftLegNode: SKShapeNode?
    private var rightLegNode: SKShapeNode?
    private var leftFlagNode: SKNode?
    private var rightFlagNode: SKNode?
    
    private var mainEngineEmitter: SKEmitterNode?
    private var leftEngineEmitter: SKEmitterNode?
    private var rightEngineEmitter: SKEmitterNode?
    
    private let physicsScale: CGFloat = 30.0
    private let viewportW: CGFloat = 600
    private let viewportH: CGFloat = 400
    
    private let landerPoly: [CGPoint] = [
        CGPoint(x: -14, y: 17),
        CGPoint(x: -17, y: 0),
        CGPoint(x: -17, y: -10),
        CGPoint(x: 17, y: -10),
        CGPoint(x: 17, y: 0),
        CGPoint(x: 14, y: 17)
    ]
    
    private let legW: CGFloat = 4
    private let legH: CGFloat = 16
    
    public override init(size: CGSize = CGSize(width: 600, height: 400)) {
        super.init(size: size)
        self.scaleMode = .resizeFill
        self.backgroundColor = SKColor(red: 0.0, green: 0.0, blue: 0.1, alpha: 1.0)
    }
    
    required init?(coder aDecoder: NSCoder) {
        return nil
    }
    
    public func updateSnapshot(_ newSnapshot: LunarLanderSnapshot) {
        self.snapshot = newSnapshot
        
        if terrainNode == nil {
            setupScene()
        }
        
        updateTerrain()
        updateLander()
        updateFlames()
    }
    
    private func setupScene() {
        removeAllChildren()
        
        createStarfield()
        
        terrainNode = SKShapeNode()
        terrainNode?.strokeColor = .clear
        terrainNode?.fillColor = SKColor(red: 0.3, green: 0.3, blue: 0.3, alpha: 1.0)
        terrainNode?.zPosition = 1
        addChild(terrainNode!)
        
        leftFlagNode = createFlag()
        leftFlagNode?.zPosition = 2
        addChild(leftFlagNode!)
        
        rightFlagNode = createFlag()
        rightFlagNode?.zPosition = 2
        addChild(rightFlagNode!)
        
        landerNode = createLanderShape()
        landerNode?.zPosition = 10
        addChild(landerNode!)
        
        leftLegNode = createLegShape()
        leftLegNode?.zPosition = 9
        addChild(leftLegNode!)
        
        rightLegNode = createLegShape()
        rightLegNode?.zPosition = 9
        addChild(rightLegNode!)
        
        mainEngineEmitter = createMainEngineEmitter()
        mainEngineEmitter?.zPosition = 8
        mainEngineEmitter?.particleBirthRate = 0
        addChild(mainEngineEmitter!)
        
        leftEngineEmitter = createSideEngineEmitter()
        leftEngineEmitter?.zPosition = 8
        leftEngineEmitter?.particleBirthRate = 0
        addChild(leftEngineEmitter!)
        
        rightEngineEmitter = createSideEngineEmitter()
        rightEngineEmitter?.zPosition = 8
        rightEngineEmitter?.particleBirthRate = 0
        addChild(rightEngineEmitter!)
    }
    
    private func createStarfield() {
        for _ in 0..<100 {
            let star = SKShapeNode(circleOfRadius: CGFloat.random(in: 0.5...1.5))
            star.fillColor = .white
            star.strokeColor = .clear
            star.position = CGPoint(
                x: CGFloat.random(in: 0...size.width),
                y: CGFloat.random(in: 0...size.height)
            )
            star.alpha = CGFloat.random(in: 0.3...1.0)
            star.zPosition = 0
            addChild(star)
        }
    }
    
    private func createLanderShape() -> SKShapeNode {
        let path = CGMutablePath()
        path.move(to: landerPoly[0])
        for i in 1..<landerPoly.count {
            path.addLine(to: landerPoly[i])
        }
        path.closeSubpath()
        
        let node = SKShapeNode(path: path)
        node.fillColor = SKColor(red: 0.5, green: 0.4, blue: 0.9, alpha: 1.0)
        node.strokeColor = SKColor(red: 0.3, green: 0.3, blue: 0.5, alpha: 1.0)
        node.lineWidth = 2
        return node
    }
    
    private func createLegShape() -> SKShapeNode {
        let path = CGMutablePath()
        let halfW = legW / 2
        let halfH = legH / 2
        
        path.addRect(CGRect(x: -halfW, y: -halfH, width: legW, height: legH))
        
        let node = SKShapeNode(path: path)
        node.fillColor = SKColor(red: 0.5, green: 0.4, blue: 0.9, alpha: 1.0)
        node.strokeColor = SKColor(red: 0.3, green: 0.3, blue: 0.5, alpha: 1.0)
        node.lineWidth = 1
        return node
    }
    
    private func createFlag() -> SKNode {
        let container = SKNode()
        
        let pole = SKShapeNode(rectOf: CGSize(width: 2, height: 50))
        pole.fillColor = .white
        pole.strokeColor = .clear
        pole.position = CGPoint(x: 0, y: 25)
        container.addChild(pole)
        
        let flagPath = CGMutablePath()
        flagPath.move(to: CGPoint(x: 0, y: 50))
        flagPath.addLine(to: CGPoint(x: 0, y: 40))
        flagPath.addLine(to: CGPoint(x: 25, y: 45))
        flagPath.closeSubpath()
        
        let flag = SKShapeNode(path: flagPath)
        flag.fillColor = SKColor(red: 0.8, green: 0.8, blue: 0.0, alpha: 1.0)
        flag.strokeColor = .clear
        container.addChild(flag)
        
        return container
    }
    
    /// Particles for engine
    private func createMainEngineEmitter() -> SKEmitterNode {
        let emitter = SKEmitterNode()
        
        emitter.particleSize = CGSize(width: 6, height: 6)
        emitter.particleBlendMode = .add
        
        emitter.particleColor = SKColor(red: 1.0, green: 0.7, blue: 0.3, alpha: 1.0)
        emitter.particleColorBlendFactor = 1.0
        emitter.particleColorSequence = SKKeyframeSequence(
            keyframeValues: [
                SKColor(red: 1.0, green: 0.9, blue: 0.5, alpha: 1.0),
                SKColor(red: 1.0, green: 0.5, blue: 0.2, alpha: 1.0),
                SKColor(red: 0.4, green: 0.2, blue: 0.2, alpha: 0.5),
                SKColor(red: 0.2, green: 0.2, blue: 0.2, alpha: 0.0),
            ],
            times: [0, 0.2, 0.6, 1.0]
        )
        
        emitter.particleLifetime = 0.13
        emitter.particleLifetimeRange = 0.05
        
        emitter.emissionAngle = -.pi / 2
        emitter.emissionAngleRange = .pi / 8
        
        emitter.particleSpeed = 110
        emitter.particleSpeedRange = 40
        
        emitter.particleBirthRate = 0
        
        emitter.particleScale = 1.0
        emitter.particleScaleSpeed = -6.0
        
        emitter.particleAlpha = 1.0
        emitter.particleAlphaSpeed = -8.0
        
        return emitter
    }
    
    /// Particles for side engines
    private func createSideEngineEmitter() -> SKEmitterNode {
        let emitter = SKEmitterNode()
        
        emitter.particleSize = CGSize(width: 4, height: 4)
        emitter.particleBlendMode = .add
        
        emitter.particleColor = SKColor(red: 1.0, green: 0.7, blue: 0.3, alpha: 1.0)
        emitter.particleColorBlendFactor = 1.0
        emitter.particleColorSequence = SKKeyframeSequence(
            keyframeValues: [
                SKColor(red: 1.0, green: 0.9, blue: 0.5, alpha: 1.0),
                SKColor(red: 1.0, green: 0.5, blue: 0.2, alpha: 1.0),
                SKColor(red: 0.4, green: 0.2, blue: 0.2, alpha: 0.5),
                SKColor(red: 0.2, green: 0.2, blue: 0.2, alpha: 0.0),
            ],
            times: [0, 0.2, 0.6, 1.0]
        )
        
        emitter.particleLifetime = 0.12
        emitter.particleLifetimeRange = 0.05
        
        emitter.emissionAngle = 0
        emitter.emissionAngleRange = .pi / 10
        
        emitter.particleSpeed = 40
        emitter.particleSpeedRange = 20
        
        emitter.particleBirthRate = 0
        
        emitter.particleScale = 1.0
        emitter.particleScaleSpeed = -8.0
        
        emitter.particleAlpha = 1.0
        emitter.particleAlphaSpeed = -10.0
        
        return emitter
    }
    
    private func updateTerrain() {
        guard let terrain = terrainNode,
              snapshot.terrainX.count > 1,
              snapshot.terrainY.count > 1 else { return }
        
        let scaleX = size.width / viewportW
        let scaleY = size.height / viewportH
        
        let path = CGMutablePath()
        
        path.move(to: CGPoint(x: 0, y: 0))
        
        for i in 0..<snapshot.terrainX.count {
            let x = CGFloat(snapshot.terrainX[i]) * physicsScale * scaleX
            let y = CGFloat(snapshot.terrainY[i]) * physicsScale * scaleY
            path.addLine(to: CGPoint(x: x, y: y))
        }
        
        path.addLine(to: CGPoint(x: size.width, y: 0))
        path.closeSubpath()
        
        terrain.path = path
        
        let helipadY = CGFloat(snapshot.helipadY) * physicsScale * scaleY
        leftFlagNode?.position = CGPoint(
            x: CGFloat(snapshot.helipadX1) * physicsScale * scaleX,
            y: helipadY
        )
        rightFlagNode?.position = CGPoint(
            x: CGFloat(snapshot.helipadX2) * physicsScale * scaleX,
            y: helipadY
        )
    }
    
    private func updateLander() {
        guard let lander = landerNode,
              let leftLeg = leftLegNode,
              let rightLeg = rightLegNode else { return }
        
        let scaleX = size.width / viewportW
        let scaleY = size.height / viewportH
        let uniformScale = min(scaleX, scaleY)
        
        let screenX = CGFloat(snapshot.x) * physicsScale * scaleX
        let screenY = CGFloat(snapshot.y) * physicsScale * scaleY
        
        lander.position = CGPoint(x: screenX, y: screenY)
        lander.zRotation = CGFloat(snapshot.angle)
        lander.setScale(uniformScale)
        
        let leftLegScreenX = CGFloat(snapshot.leftLegX) * physicsScale * scaleX
        let leftLegScreenY = CGFloat(snapshot.leftLegY) * physicsScale * scaleY
        leftLeg.position = CGPoint(x: leftLegScreenX, y: leftLegScreenY)
        leftLeg.zRotation = CGFloat(snapshot.leftLegAngle)
        leftLeg.setScale(uniformScale)
        
        let rightLegScreenX = CGFloat(snapshot.rightLegX) * physicsScale * scaleX
        let rightLegScreenY = CGFloat(snapshot.rightLegY) * physicsScale * scaleY
        rightLeg.position = CGPoint(x: rightLegScreenX, y: rightLegScreenY)
        rightLeg.zRotation = CGFloat(snapshot.rightLegAngle)
        rightLeg.setScale(uniformScale)
        
        if snapshot.gameOver {
            lander.fillColor = SKColor(red: 0.8, green: 0.2, blue: 0.2, alpha: 1.0)
            lander.strokeColor = SKColor(red: 0.5, green: 0.1, blue: 0.1, alpha: 1.0)
            leftLeg.fillColor = SKColor(red: 0.8, green: 0.2, blue: 0.2, alpha: 1.0)
            rightLeg.fillColor = SKColor(red: 0.8, green: 0.2, blue: 0.2, alpha: 1.0)
        } else {
            lander.fillColor = SKColor(red: 0.5, green: 0.4, blue: 0.9, alpha: 1.0)
            lander.strokeColor = SKColor(red: 0.3, green: 0.3, blue: 0.5, alpha: 1.0)
            
            leftLeg.fillColor = snapshot.leftLegContact ?
                SKColor(red: 0.3, green: 0.8, blue: 0.3, alpha: 1.0) :
                SKColor(red: 0.5, green: 0.4, blue: 0.9, alpha: 1.0)
            
            rightLeg.fillColor = snapshot.rightLegContact ?
                SKColor(red: 0.3, green: 0.8, blue: 0.3, alpha: 1.0) :
                SKColor(red: 0.5, green: 0.4, blue: 0.9, alpha: 1.0)
        }
    }
    
    private func updateFlames() {
        guard let mainEmitter = mainEngineEmitter,
              let leftEmitter = leftEngineEmitter,
              let rightEmitter = rightEngineEmitter,
              let lander = landerNode else { return }
        
        let scaleX = size.width / viewportW
        let scaleY = size.height / viewportH
        let uniformScale = min(scaleX, scaleY)
        
        let angle = CGFloat(snapshot.angle)
        let cosA = cos(angle)
        let sinA = sin(angle)

        let approxFPS: CGFloat = 50.0
        let powerEps: Float = 1e-4
        
        if snapshot.mainEnginePower > powerEps {
            let flameOffset: CGFloat = 12 * uniformScale
            let flameX = lander.position.x - flameOffset * sinA
            let flameY = lander.position.y - flameOffset * cosA
            mainEmitter.position = CGPoint(x: flameX, y: flameY)
            
            mainEmitter.emissionAngle = angle - .pi / 2
            
            let power = CGFloat(snapshot.mainEnginePower)
            mainEmitter.particleBirthRate = approxFPS * power
            mainEmitter.particleSpeed = 80 + 60 * power
            mainEmitter.setScale(uniformScale)
        } else {
            mainEmitter.particleBirthRate = 0
        }
        
        let sideEngineAway: CGFloat = 17 * uniformScale
        let sideEngineHeight: CGFloat = 5 * uniformScale
        
        if snapshot.sideEnginePower < -powerEps {
            let flameX = lander.position.x + sideEngineAway * cosA - sideEngineHeight * sinA
            let flameY = lander.position.y + sideEngineAway * sinA + sideEngineHeight * cosA
            leftEmitter.position = CGPoint(x: flameX, y: flameY)
            
            leftEmitter.emissionAngle = angle + .pi
            
            let power = CGFloat(abs(snapshot.sideEnginePower))
            leftEmitter.particleBirthRate = approxFPS * power
            leftEmitter.particleSpeed = 25 + 25 * power
            leftEmitter.setScale(uniformScale)
            
            rightEmitter.particleBirthRate = 0
        }
        else if snapshot.sideEnginePower > powerEps {
            let flameX = lander.position.x - sideEngineAway * cosA - sideEngineHeight * sinA
            let flameY = lander.position.y - sideEngineAway * sinA + sideEngineHeight * cosA
            rightEmitter.position = CGPoint(x: flameX, y: flameY)
            
            rightEmitter.emissionAngle = angle
            
            let power = CGFloat(snapshot.sideEnginePower)
            rightEmitter.particleBirthRate = approxFPS * power
            rightEmitter.particleSpeed = 25 + 25 * power
            rightEmitter.setScale(uniformScale)
            
            leftEmitter.particleBirthRate = 0
        } else {
            leftEmitter.particleBirthRate = 0
            rightEmitter.particleBirthRate = 0
        }
    }
}

#Preview {
    let sampleSnapshot = LunarLanderSnapshot(
        x: 10, y: 10, angle: 0.1,
        leftLegX: 10.667, leftLegY: 9.4, leftLegAngle: 0.4,
        rightLegX: 9.333, rightLegY: 9.4, rightLegAngle: -0.4,
        leftLegContact: false, rightLegContact: false,
        mainEnginePower: 1.0, sideEnginePower: 0,
        terrainX: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        terrainY: [2, 2.5, 3, 3.33, 3.33, 3.33, 3.33, 3, 2.5, 2, 2],
        helipadX1: 8, helipadX2: 12, helipadY: 3.33,
        gameOver: false
    )
    
    LunarLanderView(snapshot: sampleSnapshot)
        .frame(width: 600, height: 400)
}
#endif
