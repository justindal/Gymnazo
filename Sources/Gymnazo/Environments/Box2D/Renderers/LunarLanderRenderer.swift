//
//  LunarLanderRenderer.swift
//

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
            Text("LunarLander rendering requires iOS 17+ / macOS 14+")
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
    private var mainFlameNode: SKShapeNode?
    private var leftFlameNode: SKShapeNode?
    private var rightFlameNode: SKShapeNode?
    
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
    
    private let legAway: CGFloat = 20
    private let legDown: CGFloat = 18
    private let legW: CGFloat = 3
    private let legH: CGFloat = 12
    
    public override init(size: CGSize = CGSize(width: 600, height: 400)) {
        super.init(size: size)
        self.scaleMode = .resizeFill
        self.backgroundColor = SKColor(red: 0.0, green: 0.0, blue: 0.1, alpha: 1.0)
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
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
        
        mainFlameNode = createMainFlame()
        mainFlameNode?.zPosition = 8
        mainFlameNode?.isHidden = true
        addChild(mainFlameNode!)
        
        leftFlameNode = createSideFlame()
        leftFlameNode?.zPosition = 8
        leftFlameNode?.isHidden = true
        addChild(leftFlameNode!)
        
        rightFlameNode = createSideFlame()
        rightFlameNode?.zPosition = 8
        rightFlameNode?.isHidden = true
        addChild(rightFlameNode!)
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
        let scaledPoly = landerPoly.map { CGPoint(x: $0.x * 0.8, y: $0.y * 0.8) }
        
        path.move(to: scaledPoly[0])
        for i in 1..<scaledPoly.count {
            path.addLine(to: scaledPoly[i])
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
    
    private func createMainFlame() -> SKShapeNode {
        let path = CGMutablePath()
        path.move(to: CGPoint(x: -8, y: 0))
        path.addLine(to: CGPoint(x: 0, y: -30))
        path.addLine(to: CGPoint(x: 8, y: 0))
        path.closeSubpath()
        
        let node = SKShapeNode(path: path)
        node.fillColor = SKColor(red: 1.0, green: 0.6, blue: 0.2, alpha: 1.0)
        node.strokeColor = SKColor(red: 1.0, green: 0.9, blue: 0.5, alpha: 1.0)
        node.lineWidth = 2
        return node
    }
    
    private func createSideFlame() -> SKShapeNode {
        let path = CGMutablePath()
        path.move(to: CGPoint(x: 0, y: -4))
        path.addLine(to: CGPoint(x: 15, y: 0))
        path.addLine(to: CGPoint(x: 0, y: 4))
        path.closeSubpath()
        
        let node = SKShapeNode(path: path)
        node.fillColor = SKColor(red: 1.0, green: 0.6, blue: 0.2, alpha: 1.0)
        node.strokeColor = SKColor(red: 1.0, green: 0.9, blue: 0.5, alpha: 1.0)
        node.lineWidth = 1
        return node
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
        
        let screenX = CGFloat(snapshot.x) * physicsScale * scaleX
        let screenY = CGFloat(snapshot.y) * physicsScale * scaleY
        
        lander.position = CGPoint(x: screenX, y: screenY)
        lander.zRotation = -CGFloat(snapshot.angle)
        
        let angle = CGFloat(snapshot.angle)
        let cosA = cos(angle)
        let sinA = sin(angle)
        
        let legAwayScaled = legAway * 0.8 * scaleX
        let legDownScaled = legDown * 0.8 * scaleY
        
        let leftOffsetX = -legAwayScaled * cosA + legDownScaled * sinA
        let leftOffsetY = -legAwayScaled * sinA - legDownScaled * cosA
        leftLeg.position = CGPoint(x: screenX + leftOffsetX, y: screenY + leftOffsetY)
        leftLeg.zRotation = -angle
        leftLeg.fillColor = snapshot.leftLegContact ?
            SKColor(red: 0.3, green: 0.8, blue: 0.3, alpha: 1.0) :
            SKColor(red: 0.5, green: 0.4, blue: 0.9, alpha: 1.0)
        
        let rightOffsetX = legAwayScaled * cosA + legDownScaled * sinA
        let rightOffsetY = legAwayScaled * sinA - legDownScaled * cosA
        rightLeg.position = CGPoint(x: screenX + rightOffsetX, y: screenY + rightOffsetY)
        rightLeg.zRotation = -angle
        rightLeg.fillColor = snapshot.rightLegContact ?
            SKColor(red: 0.3, green: 0.8, blue: 0.3, alpha: 1.0) :
            SKColor(red: 0.5, green: 0.4, blue: 0.9, alpha: 1.0)
    }
    
    private func updateFlames() {
        guard let mainFlame = mainFlameNode,
              let leftFlame = leftFlameNode,
              let rightFlame = rightFlameNode,
              let lander = landerNode else { return }
        
        let angle = CGFloat(snapshot.angle)
        let cosA = cos(angle)
        let sinA = sin(angle)
        
        if snapshot.mainEnginePower > 0 {
            mainFlame.isHidden = false
            
            let flameOffset: CGFloat = 15
            let flameX = lander.position.x + flameOffset * sinA
            let flameY = lander.position.y - flameOffset * cosA
            mainFlame.position = CGPoint(x: flameX, y: flameY)
            mainFlame.zRotation = -angle
            
            let scale = 0.5 + CGFloat(snapshot.mainEnginePower) * 0.5
            mainFlame.setScale(scale)
        } else {
            mainFlame.isHidden = true
        }
        
        let sideEngineOffset: CGFloat = 12
        let sideEngineHeight: CGFloat = 10
        
        if snapshot.sideEnginePower < 0 {
            leftFlame.isHidden = false
            rightFlame.isHidden = true
            
            let flameX = lander.position.x + sideEngineOffset * cosA + sideEngineHeight * sinA
            let flameY = lander.position.y + sideEngineOffset * sinA - sideEngineHeight * cosA
            leftFlame.position = CGPoint(x: flameX, y: flameY)
            leftFlame.zRotation = -angle
            leftFlame.xScale = -1
        } else if snapshot.sideEnginePower > 0 {
            leftFlame.isHidden = true
            rightFlame.isHidden = false
            
            let flameX = lander.position.x - sideEngineOffset * cosA + sideEngineHeight * sinA
            let flameY = lander.position.y - sideEngineOffset * sinA - sideEngineHeight * cosA
            rightFlame.position = CGPoint(x: flameX, y: flameY)
            rightFlame.zRotation = -angle
            rightFlame.xScale = 1
        } else {
            leftFlame.isHidden = true
            rightFlame.isHidden = true
        }
    }
}

#Preview {
    let sampleSnapshot = LunarLanderSnapshot(
        x: 10, y: 10, angle: 0.1,
        leftLegContact: false, rightLegContact: false,
        mainEnginePower: 0.8, sideEnginePower: 0,
        terrainX: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        terrainY: [2, 2.5, 3, 3.33, 3.33, 3.33, 3.33, 3, 2.5, 2, 2],
        helipadX1: 8, helipadX2: 12, helipadY: 3.33,
        gameOver: false
    )
    
    LunarLanderView(snapshot: sampleSnapshot)
        .frame(width: 600, height: 400)
}
#endif
