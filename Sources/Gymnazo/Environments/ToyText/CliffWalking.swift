import Foundation
import MLX
#if canImport(SwiftUI)
import SwiftUI
import CoreGraphics
#if os(macOS)
import AppKit
#elseif os(iOS) || os(tvOS) || os(visionOS)
import UIKit
#endif
#endif
#if canImport(PlaygroundSupport)
import PlaygroundSupport
#endif

/// Cliff walking involves crossing a gridworld from start to goal while avoiding falling off a cliff.
///
/// ## Description
///
/// The game starts with the player at location [3, 0] of the 4x12 grid world with the
/// goal located at [3, 11]. If the player reaches the goal the episode ends.
///
/// A cliff runs along [3, 1..10]. If the player moves to a cliff location it
/// returns to the start location with a -100 reward penalty.
///
/// The player makes moves until they reach the goal.
///
/// ```
/// o  o  o  o  o  o  o  o  o  o  o  o
/// o  o  o  o  o  o  o  o  o  o  o  o
/// o  o  o  o  o  o  o  o  o  o  o  o
/// S  C  C  C  C  C  C  C  C  C  C  G
/// ```
///
/// Adapted from Example 6.6 (page 132) from Reinforcement Learning: An Introduction
/// by Sutton and Barto.
///
/// ## Action Space
///
/// The action is an `Int` in the range `{0, 3}`:
///
/// | Action | Direction |
/// |--------|-----------|
/// | 0      | Up        |
/// | 1      | Right     |
/// | 2      | Down      |
/// | 3      | Left      |
///
/// ## Observation Space
///
/// There are 48 discrete states (4 rows x 12 columns).
/// The observation is encoded as: `row * 12 + col`
///
/// The starting position is state 36 (location [3, 0]).
/// The goal position is state 47 (location [3, 11]).
///
/// ## Starting State
///
/// The episode always starts with the player at state 36 (location [3, 0]).
///
/// ## Rewards
///
/// - **-1** per step
/// - **-100** for stepping on the cliff (then returns to start)
///
/// ## Episode End
///
/// The episode terminates when the player reaches state 47 (location [3, 11]).
///
/// ## Arguments
///
/// - `render_mode`: The render mode (`"human"`, `"ansi"`, or `"rgb_array"`).
/// - `isSlippery`: If `true`, the player has 1/3 probability each for intended direction, left, right.
///
/// ## Version History
///
/// - v1: initial implementation
public final class CliffWalking: Env {
    private static let numRows = 4
    private static let numCols = 12
    private static let numStates = numRows * numCols
    private static let numActions = 4
    
    public static let startState = 36
    public static let goalState = 47
    
    private static let positionDeltas: [(dRow: Int, dCol: Int)] = [
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1)
    ]
    
    private func toInt(_ action: MLXArray) -> Int {
        Int(action.item(Int32.self))
    }
    
    private func toMLX(_ state: Int) -> MLXArray {
        MLXArray([Int32(state)])
    }
    
    public static var metadata: [String: Any] {
        [
            "render_modes": ["human", "ansi", "rgb_array"],
            "render_fps": 4,
        ]
    }
    
    public let actionSpace: any Space
    public let observationSpace: any Space
    public var spec: EnvSpec?
    public var renderMode: RenderMode?
    
    private let isSlippery: Bool
    private var cliff: [[Bool]]
    
    private typealias Transition = (prob: Double, nextState: Int, reward: Double, terminated: Bool)
    private var P: [[Int: [Transition]]]
    
    private var s: Int = 0
    private var lastAction: Int?
    
    private var _key: MLXArray?
    
#if canImport(SwiftUI)
    private var lastRGBFrame: CGImage?
#endif
    
    public init(
        renderMode: RenderMode? = nil,
        isSlippery: Bool = false
    ) {
        self.renderMode = renderMode
        self.isSlippery = isSlippery
        
        self.actionSpace = Discrete(n: Self.numActions)
        self.observationSpace = Discrete(n: Self.numStates)
        
        self.cliff = Array(repeating: Array(repeating: false, count: Self.numCols), count: Self.numRows)
        for col in 1..<(Self.numCols - 1) {
            cliff[3][col] = true
        }
        
        self.P = Array(repeating: [:], count: Self.numStates)
        
        for state in 0..<Self.numStates {
            let (row, col) = Self.stateToPosition(state)
            P[state] = [:]
            
            for action in 0..<Self.numActions {
                P[state][action] = calculateTransitionProb(row: row, col: col, action: action)
            }
        }
    }
    
    public static func positionToState(row: Int, col: Int) -> Int {
        row * numCols + col
    }
    
    public static func stateToPosition(_ state: Int) -> (row: Int, col: Int) {
        let row = state / numCols
        let col = state % numCols
        return (row, col)
    }
    
    private func limitCoordinates(row: Int, col: Int) -> (row: Int, col: Int) {
        let newRow = max(0, min(row, Self.numRows - 1))
        let newCol = max(0, min(col, Self.numCols - 1))
        return (newRow, newCol)
    }
    
    private func calculateTransitionProb(row: Int, col: Int, action: Int) -> [Transition] {
        let actions: [Int]
        if isSlippery {
            actions = [(action + 3) % 4, action, (action + 1) % 4]
        } else {
            actions = [action]
        }
        
        var outcomes: [Transition] = []
        let prob = 1.0 / Double(actions.count)
        
        for act in actions {
            let delta = Self.positionDeltas[act]
            let newRow = row + delta.dRow
            let newCol = col + delta.dCol
            let (limitedRow, limitedCol) = limitCoordinates(row: newRow, col: newCol)
            let newState = Self.positionToState(row: limitedRow, col: limitedCol)
            
            if cliff[limitedRow][limitedCol] {
                outcomes.append((prob, Self.startState, -100.0, false))
            } else {
                let isTerminated = newState == Self.goalState
                outcomes.append((prob, newState, -1.0, isTerminated))
            }
        }
        
        return outcomes
    }
    
    private func prepareKey(with seed: UInt64?) throws -> MLXArray {
        if let seed {
            _key = MLX.key(seed)
        } else if _key == nil {
            _key = MLX.key(UInt64.random(in: 0...UInt64.max))
        }
        guard let key = _key else {
            throw GymnazoError.invalidState("Failed to initialize RNG key")
        }
        return key
    }
    
    public func reset(
        seed: UInt64? = nil,
        options: EnvOptions? = nil
    ) throws -> Reset {
        _ = try prepareKey(with: seed)
        
        s = Self.startState
        lastAction = nil
        
        return Reset(obs: toMLX(s), info: ["prob": 1.0])
    }
    
    public func step(_ action: MLXArray) throws -> Step {
        let a = toInt(action)
        guard let transitions = P[s][a], !transitions.isEmpty else {
            throw GymnazoError.invalidState("Invalid state or action")
        }
        
        guard let key = _key else {
            throw GymnazoError.invalidState("Missing RNG key")
        }
        let (sampleKey, nextKey) = MLX.split(key: key)
        _key = nextKey
        
        let probs = transitions.map { Float($0.prob) }
        let epsilon = MLXArray(1e-9, dtype: .float32)
        let logits = MLX.log(MLXArray(probs) + epsilon)
        let i = Int(MLX.categorical(logits, key: sampleKey).item(Int32.self))
        
        let (p, newState, reward, terminated) = transitions[i]
        
        s = newState
        lastAction = a
        
        return Step(
            obs: toMLX(s),
            reward: reward,
            terminated: terminated,
            truncated: false,
            info: ["prob": .double(p)]
        )
    }
    
    public func renderAnsi() -> String {
        _renderText()
    }
    
    @discardableResult
    public func render() throws -> RenderOutput? {
        guard let mode = renderMode else {
            if let specId = spec?.id {
                print("[Gymnazo] render() called without renderMode. Set renderMode when creating \(specId).")
            }
            return nil
        }
        
        switch mode {
        case .ansi:
            return .ansi(_renderText())
        case .human:
#if canImport(SwiftUI)
            return .other(currentSnapshot)
#else
            return nil
#endif
        case .rgbArray:
            return nil
        case .statePixels:
            print("[Gymnazo] Unsupported renderMode \(mode.rawValue).")
            return nil
        }
    }
    
    private func _renderText() -> String {
        var lines: [String] = []
        
        for row in 0..<Self.numRows {
            var rowStr = ""
            for col in 0..<Self.numCols {
                let state = Self.positionToState(row: row, col: col)
                let cell: String
                
                if state == s {
                    cell = " x "
                } else if state == Self.goalState {
                    cell = " G "
                } else if state == Self.startState {
                    cell = " S "
                } else if cliff[row][col] {
                    cell = " C "
                } else {
                    cell = " o "
                }
                
                rowStr += cell
            }
            lines.append(rowStr)
        }
        
        var result = lines.joined(separator: "\n")
        result += "\nLegend: x=Player, S=Start, G=Goal, C=Cliff, o=Safe"
        
        return result
    }
    
#if canImport(SwiftUI)
    @MainActor
    private static func renderGUI(snapshot: CliffWalkingRenderSnapshot, mode: RenderMode) -> CGImage? {
        let view = CliffWalkingCanvasView(snapshot: snapshot)
        
        switch mode {
        case .human:
#if canImport(PlaygroundSupport)
            PlaygroundPage.current.setLiveView(view)
#else
            print("[Gymnazo] SwiftUI Canvas available via CliffWalkingCanvasView; integrate it into your app UI.")
#endif
            return nil
        case .rgbArray:
            if #available(macOS 13.0, iOS 16.0, *) {
                let renderer = ImageRenderer(content: view)
#if os(macOS)
                renderer.scale = NSScreen.main?.backingScaleFactor ?? 2.0
#else
                renderer.scale = UIScreen.main.scale
#endif
                return renderer.cgImage
            } else {
                print("[Gymnazo] rgb_array rendering requires macOS 13/iOS 16.")
                return nil
            }
        case .ansi, .statePixels:
            return nil
        }
    }
    
    public var latestRGBFrame: CGImage? {
        lastRGBFrame
    }

    @MainActor
    public func renderRGBArray() -> CGImage? {
        let image = Self.renderGUI(snapshot: currentSnapshot, mode: .rgbArray)
        lastRGBFrame = image
        return image
    }
    
    public var currentSnapshot: CliffWalkingRenderSnapshot {
        let (row, col) = Self.stateToPosition(s)
        return CliffWalkingRenderSnapshot(
            playerRow: row,
            playerCol: col,
            lastAction: lastAction
        )
    }
    
    @MainActor
    public func humanView() -> CliffWalkingCanvasView {
        CliffWalkingCanvasView(snapshot: currentSnapshot)
    }
#endif
}

#if canImport(SwiftUI)
/// CliffWalking SwiftUI Snapshot
public struct CliffWalkingRenderSnapshot: Sendable, Equatable {
    public let playerRow: Int
    public let playerCol: Int
    public let lastAction: Int?
}

/// SwiftUI Canvas view
public struct CliffWalkingCanvasView: View {
    public let snapshot: CliffWalkingRenderSnapshot
    
    public init(snapshot: CliffWalkingRenderSnapshot) {
        self.snapshot = snapshot
    }
    
    private let cellWidth: CGFloat = 50
    private let cellHeight: CGFloat = 50
    private let numRows = 4
    private let numCols = 12
    
    public var body: some View {
        Canvas { context, size in
            drawBackground(context: &context, size: size)
            drawGrid(context: &context)
            drawCliff(context: &context)
            drawStart(context: &context)
            drawGoal(context: &context)
            drawPlayer(context: &context)
        }
        .frame(width: CGFloat(numCols) * cellWidth + 40, height: CGFloat(numRows) * cellHeight + 40)
    }
    
    private func cellOrigin(row: Int, col: Int) -> CGPoint {
        CGPoint(x: 20 + CGFloat(col) * cellWidth, y: 20 + CGFloat(row) * cellHeight)
    }
    
    private func cellCenter(row: Int, col: Int) -> CGPoint {
        let origin = cellOrigin(row: row, col: col)
        return CGPoint(x: origin.x + cellWidth / 2, y: origin.y + cellHeight / 2)
    }
    
    private func drawBackground(context: inout GraphicsContext, size: CGSize) {
        let bgColor = Color(red: 0.4, green: 0.6, blue: 0.4)
        context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(bgColor))
    }
    
    private func drawGrid(context: inout GraphicsContext) {
        for row in 0..<numRows {
            for col in 0..<numCols {
                let origin = cellOrigin(row: row, col: col)
                let rect = CGRect(origin: origin, size: CGSize(width: cellWidth, height: cellHeight))
                
                let isAlternate = (row + col) % 2 == 0
                let cellColor = isAlternate ?
                    Color(red: 0.5, green: 0.7, blue: 0.5) :
                    Color(red: 0.45, green: 0.65, blue: 0.45)
                
                context.fill(Path(rect), with: .color(cellColor))
                context.stroke(Path(rect), with: .color(Color(white: 0.3, opacity: 0.3)), style: StrokeStyle(lineWidth: 1))
            }
        }
    }
    
    private func drawCliff(context: inout GraphicsContext) {
        for col in 1..<(numCols - 1) {
            let origin = cellOrigin(row: 3, col: col)
            let rect = CGRect(origin: origin, size: CGSize(width: cellWidth, height: cellHeight))
            
            let cliffColor = Color(red: 0.3, green: 0.15, blue: 0.1)
            context.fill(Path(rect), with: .color(cliffColor))
            
            context.drawLayer { layerContext in
                let stripeSpacing: CGFloat = 8
                var stripePath = Path()
                for i in stride(from: 0, to: cellWidth + cellHeight, by: stripeSpacing) {
                    stripePath.move(to: CGPoint(x: origin.x + i, y: origin.y))
                    stripePath.addLine(to: CGPoint(x: origin.x, y: origin.y + i))
                }
                layerContext.clip(to: Path(rect))
                layerContext.stroke(stripePath, with: .color(Color.red.opacity(0.4)), style: StrokeStyle(lineWidth: 2))
            }
        }
        
        for col in 1..<(numCols - 1) {
            let center = cellCenter(row: 3, col: col)
            context.draw(
                Text("!")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundColor(.yellow),
                at: center
            )
        }
    }
    
    private func drawStart(context: inout GraphicsContext) {
        let center = cellCenter(row: 3, col: 0)
        
        let flagPoleHeight: CGFloat = 30
        var polePath = Path()
        polePath.move(to: CGPoint(x: center.x - 10, y: center.y + 15))
        polePath.addLine(to: CGPoint(x: center.x - 10, y: center.y + 15 - flagPoleHeight))
        context.stroke(polePath, with: .color(.white), style: StrokeStyle(lineWidth: 3))
        
        var flagPath = Path()
        flagPath.move(to: CGPoint(x: center.x - 10, y: center.y + 15 - flagPoleHeight))
        flagPath.addLine(to: CGPoint(x: center.x + 10, y: center.y + 15 - flagPoleHeight + 8))
        flagPath.addLine(to: CGPoint(x: center.x - 10, y: center.y + 15 - flagPoleHeight + 16))
        flagPath.closeSubpath()
        context.fill(flagPath, with: .color(.green))
        context.stroke(flagPath, with: .color(.white), style: StrokeStyle(lineWidth: 1))
        
        context.draw(
            Text("S")
                .font(.system(size: 12, weight: .bold))
                .foregroundColor(.white),
            at: CGPoint(x: center.x, y: center.y + 10)
        )
    }
    
    private func drawGoal(context: inout GraphicsContext) {
        let center = cellCenter(row: 3, col: 11)
        
        let starRadius: CGFloat = 18
        let innerRadius: CGFloat = 8
        let points = 5
        
        var starPath = Path()
        for i in 0..<(points * 2) {
            let radius = i % 2 == 0 ? starRadius : innerRadius
            let angle = Double(i) * .pi / Double(points) - .pi / 2
            let point = CGPoint(
                x: center.x + CGFloat(cos(angle)) * radius,
                y: center.y + CGFloat(sin(angle)) * radius
            )
            if i == 0 {
                starPath.move(to: point)
            } else {
                starPath.addLine(to: point)
            }
        }
        starPath.closeSubpath()
        
        context.fill(starPath, with: .color(.yellow))
        context.stroke(starPath, with: .color(.orange), style: StrokeStyle(lineWidth: 2))
        
        context.draw(
            Text("G")
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(.orange),
            at: center
        )
    }
    
    private func drawPlayer(context: inout GraphicsContext) {
        let center = cellCenter(row: snapshot.playerRow, col: snapshot.playerCol)
        
        let bodyRadius: CGFloat = 14
        let bodyPath = Path(ellipseIn: CGRect(
            x: center.x - bodyRadius,
            y: center.y - bodyRadius,
            width: bodyRadius * 2,
            height: bodyRadius * 2
        ))
        context.fill(bodyPath, with: .color(.blue))
        context.stroke(bodyPath, with: .color(.white), style: StrokeStyle(lineWidth: 2))
        
        if let action = snapshot.lastAction {
            let arrowLength: CGFloat = 8
            let arrowWidth: CGFloat = 6
            
            let angle: Double
            switch action {
            case 0: angle = -.pi / 2
            case 1: angle = 0
            case 2: angle = .pi / 2
            case 3: angle = .pi
            default: angle = 0
            }
            
            let tipX = center.x + cos(angle) * (bodyRadius + 2)
            let tipY = center.y + sin(angle) * (bodyRadius + 2)
            
            var arrowPath = Path()
            arrowPath.move(to: CGPoint(x: tipX, y: tipY))
            arrowPath.addLine(to: CGPoint(
                x: tipX - cos(angle) * arrowLength + cos(angle + .pi/2) * arrowWidth/2,
                y: tipY - sin(angle) * arrowLength + sin(angle + .pi/2) * arrowWidth/2
            ))
            arrowPath.addLine(to: CGPoint(
                x: tipX - cos(angle) * arrowLength - cos(angle + .pi/2) * arrowWidth/2,
                y: tipY - sin(angle) * arrowLength - sin(angle + .pi/2) * arrowWidth/2
            ))
            arrowPath.closeSubpath()
            
            context.fill(arrowPath, with: .color(.white))
        }
    }
}
#endif

