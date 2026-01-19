#if canImport(SwiftUI) && canImport(SpriteKit)
import SwiftUI
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
            SwiftUI.Text("Unsupported")
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
        return nil
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
