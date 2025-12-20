//
// Taxi.swift
//

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

/// The Taxi Problem involves navigating to passengers in a grid world,
/// picking them up and dropping them off at one of four locations.
///
/// ## Description
///
/// There are four designated pick-up and drop-off locations (Red, Green, Yellow and Blue)
/// in the 5x5 grid world. The taxi starts off at a random square and the passenger at one
/// of the designated locations.
///
/// The goal is to move the taxi to the passenger's location, pick up the passenger,
/// move to the passenger's desired destination, and drop off the passenger.
/// Once the passenger is dropped off, the episode ends.
///
/// Map:
/// ```
/// +---------+
/// |R: | : :G|
/// | : | : : |
/// | : : : : |
/// | | : | : |
/// |Y| : |B: |
/// +---------+
/// ```
///
/// ## Action Space
///
/// The action is an `Int` in the range `{0, 5}`:
///
/// | Action | Meaning   |
/// |--------|-----------|
/// | 0      | South     |
/// | 1      | North     |
/// | 2      | East      |
/// | 3      | West      |
/// | 4      | Pickup    |
/// | 5      | Dropoff   |
///
/// ## Observation Space
///
/// There are 500 discrete states encoded as:
/// `((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination`
///
/// Passenger locations: 0=Red, 1=Green, 2=Yellow, 3=Blue, 4=In taxi
///
/// Destinations: 0=Red, 1=Green, 2=Yellow, 3=Blue
///
/// ## Starting State
///
/// The initial state is sampled uniformly from states where the passenger
/// is neither at their destination nor inside the taxi.
///
/// ## Rewards
///
/// - **-1** per step (time penalty)
/// - **+20** successful dropoff at destination
/// - **-10** illegal pickup or dropoff
///
/// ## Episode End
///
/// The episode ends when the taxi successfully drops off the passenger at the destination.
///
/// ## Arguments
///
/// - `render_mode`: The render mode (`"human"`, `"ansi"`, or `"rgb_array"`).
/// - `isRainy`: If `true`, movement has 80% intended direction, 10% slip left/right.
/// - `ficklePassenger`: If `true`, 30% chance passenger changes destination after first pickup.
///
/// ## Version History
///
/// - v1: initial
public final class Taxi: Env {
    private static let map: [String] = [
        "+---------+",
        "|R: | : :G|",
        "| : | : : |",
        "| : : : : |",
        "| | : | : |",
        "|Y| : |B: |",
        "+---------+",
    ]
    
    public static let locs: [(row: Int, col: Int)] = [(0, 0), (0, 4), (4, 0), (4, 3)]
    public static let locColors: [String] = ["R", "G", "Y", "B"]
    
    public static var metadata: [String: Any] {
        [
            "render_modes": ["human", "ansi", "rgb_array"],
            "render_fps": 4,
        ]
    }
    
    public typealias Observation = Int
    public typealias Action = Int
    public typealias ObservationSpace = Discrete
    public typealias ActionSpace = Discrete
    
    public let action_space: Discrete
    public let observation_space: Discrete
    public var spec: EnvSpec?
    public var render_mode: String?
    
    private let desc: [[UInt8]]
    private let maxRow: Int = 4
    private let maxCol: Int = 4
    private let numStates: Int = 500
    private let numActions: Int = 6
    
    private let isRainy: Bool
    private let ficklePassenger: Bool
    
    private typealias Transition = (prob: Double, nextState: Int, reward: Double, terminated: Bool)
    private var P: [[Int: [Transition]]]
    private var initialStateDistrib: [Double]
    
    private var s: Int = 0
    private var lastAction: Int?
    private var taxiOrientation: Int = 0
    private var fickleStep: Bool = false
    
    private var _key: MLXArray?
    
#if canImport(SwiftUI)
    private var lastRGBFrame: CGImage?
#endif
    
    public init(
        render_mode: String? = nil,
        isRainy: Bool = false,
        ficklePassenger: Bool = false
    ) {
        self.render_mode = render_mode
        self.isRainy = isRainy
        self.ficklePassenger = ficklePassenger
        
        self.desc = Self.map.map { Array($0.utf8) }
        
        self.action_space = Discrete(n: numActions)
        self.observation_space = Discrete(n: numStates)
        
        self.P = Array(repeating: [:], count: numStates)
        self.initialStateDistrib = Array(repeating: 0.0, count: numStates)
        
        for row in 0..<5 {
            for col in 0..<5 {
                for passIdx in 0..<5 {
                    for destIdx in 0..<4 {
                        let state = Self.encode(taxiRow: row, taxiCol: col, passLoc: passIdx, destIdx: destIdx)
                        
                        if passIdx < 4 && passIdx != destIdx {
                            initialStateDistrib[state] = 1.0
                        }
                        
                        P[state] = [:]
                        for action in 0..<numActions {
                            if isRainy {
                                P[state][action] = buildRainyTransitions(
                                    row: row, col: col, passIdx: passIdx, destIdx: destIdx, action: action
                                )
                            } else {
                                P[state][action] = buildDryTransitions(
                                    row: row, col: col, passIdx: passIdx, destIdx: destIdx, action: action
                                )
                            }
                        }
                    }
                }
            }
        }
        
        let total = initialStateDistrib.reduce(0, +)
        if total > 0 {
            initialStateDistrib = initialStateDistrib.map { $0 / total }
        }
    }
    
    public static func encode(taxiRow: Int, taxiCol: Int, passLoc: Int, destIdx: Int) -> Int {
        var i = taxiRow
        i *= 5
        i += taxiCol
        i *= 5
        i += passLoc
        i *= 4
        i += destIdx
        return i
    }
    
    public static func decode(_ state: Int) -> (taxiRow: Int, taxiCol: Int, passLoc: Int, destIdx: Int) {
        var i = state
        let destIdx = i % 4
        i = i / 4
        let passLoc = i % 5
        i = i / 5
        let taxiCol = i % 5
        i = i / 5
        let taxiRow = i
        return (taxiRow, taxiCol, passLoc, destIdx)
    }
    
    private func pickup(taxiLoc: (Int, Int), passIdx: Int, reward: Double) -> (newPassIdx: Int, newReward: Double) {
        if passIdx < 4 && taxiLoc == Self.locs[passIdx] {
            return (4, reward)
        } else {
            return (passIdx, -10.0)
        }
    }
    
    private func dropoff(taxiLoc: (Int, Int), passIdx: Int, destIdx: Int, defaultReward: Double) -> (newPassIdx: Int, newReward: Double, terminated: Bool) {
        if taxiLoc == Self.locs[destIdx] && passIdx == 4 {
            return (destIdx, 20.0, true)
        } else if Self.locs.contains(where: { $0 == taxiLoc }) && passIdx == 4 {
            let newPassIdx = Self.locs.firstIndex(where: { $0 == taxiLoc })!
            return (newPassIdx, defaultReward, false)
        } else {
            return (passIdx, -10.0, false)
        }
    }
    
    private func canMoveEast(row: Int, col: Int) -> Bool {
        let charIndex = 2 * col + 2
        return desc[1 + row][charIndex] == UInt8(ascii: ":")
    }
    
    private func canMoveWest(row: Int, col: Int) -> Bool {
        let charIndex = 2 * col
        return desc[1 + row][charIndex] == UInt8(ascii: ":")
    }
    
    private func buildDryTransitions(row: Int, col: Int, passIdx: Int, destIdx: Int, action: Int) -> [Transition] {
        let taxiLoc = (row, col)
        var newRow = row
        var newCol = col
        var newPassIdx = passIdx
        var reward: Double = -1.0
        var terminated = false
        
        switch action {
        case 0:
            newRow = min(row + 1, maxRow)
        case 1:
            newRow = max(row - 1, 0)
        case 2:
            if canMoveEast(row: row, col: col) {
                newCol = min(col + 1, maxCol)
            }
        case 3:
            if canMoveWest(row: row, col: col) {
                newCol = max(col - 1, 0)
            }
        case 4:
            let result = pickup(taxiLoc: taxiLoc, passIdx: newPassIdx, reward: reward)
            newPassIdx = result.newPassIdx
            reward = result.newReward
        case 5:
            let result = dropoff(taxiLoc: taxiLoc, passIdx: newPassIdx, destIdx: destIdx, defaultReward: reward)
            newPassIdx = result.newPassIdx
            reward = result.newReward
            terminated = result.terminated
        default:
            break
        }
        
        let newState = Self.encode(taxiRow: newRow, taxiCol: newCol, passLoc: newPassIdx, destIdx: destIdx)
        return [(1.0, newState, reward, terminated)]
    }
    
    private func calcNewPosition(row: Int, col: Int, movement: (Int, Int), checkEast: Bool) -> (Int, Int) {
        let (dr, dc) = movement
        let newRow = max(0, min(row + dr, maxRow))
        let newCol = max(0, min(col + dc, maxCol))
        
        if dc > 0 {
            if canMoveEast(row: row, col: col) {
                return (newRow, newCol)
            }
        } else if dc < 0 {
            if canMoveWest(row: row, col: col) {
                return (newRow, newCol)
            }
        } else {
            return (newRow, newCol)
        }
        
        return (row, col)
    }
    
    private func buildRainyTransitions(row: Int, col: Int, passIdx: Int, destIdx: Int, action: Int) -> [Transition] {
        let taxiLoc = (row, col)
        var newRow = row
        var newCol = col
        var newPassIdx = passIdx
        var reward: Double = -1.0
        var terminated = false
        
        let moves: [Int: ((Int, Int), (Int, Int), (Int, Int))] = [
            0: ((1, 0), (0, -1), (0, 1)),
            1: ((-1, 0), (0, -1), (0, 1)),
            2: ((0, 1), (1, 0), (-1, 0)),
            3: ((0, -1), (1, 0), (-1, 0)),
        ]
        
        if action <= 3 {
            let canMove: Bool
            switch action {
            case 0, 1:
                canMove = true
            case 2:
                canMove = canMoveEast(row: row, col: col)
            case 3:
                canMove = canMoveWest(row: row, col: col)
            default:
                canMove = false
            }
            
            if canMove, let (intended, left, right) = moves[action] {
                let (dr, dc) = intended
                newRow = max(0, min(row + dr, maxRow))
                newCol = max(0, min(col + dc, maxCol))
                
                let leftPos = calcNewPosition(row: row, col: col, movement: left, checkEast: left.1 > 0)
                let rightPos = calcNewPosition(row: row, col: col, movement: right, checkEast: right.1 > 0)
                
                let intendedState = Self.encode(taxiRow: newRow, taxiCol: newCol, passLoc: newPassIdx, destIdx: destIdx)
                let leftState = Self.encode(taxiRow: leftPos.0, taxiCol: leftPos.1, passLoc: newPassIdx, destIdx: destIdx)
                let rightState = Self.encode(taxiRow: rightPos.0, taxiCol: rightPos.1, passLoc: newPassIdx, destIdx: destIdx)
                
                return [
                    (0.8, intendedState, -1.0, false),
                    (0.1, leftState, -1.0, false),
                    (0.1, rightState, -1.0, false)
                ]
            } else {
                let state = Self.encode(taxiRow: row, taxiCol: col, passLoc: newPassIdx, destIdx: destIdx)
                return [(1.0, state, -1.0, false)]
            }
        } else if action == 4 {
            let result = pickup(taxiLoc: taxiLoc, passIdx: newPassIdx, reward: reward)
            newPassIdx = result.newPassIdx
            reward = result.newReward
        } else if action == 5 {
            let result = dropoff(taxiLoc: taxiLoc, passIdx: newPassIdx, destIdx: destIdx, defaultReward: reward)
            newPassIdx = result.newPassIdx
            reward = result.newReward
            terminated = result.terminated
        }
        
        let newState = Self.encode(taxiRow: newRow, taxiCol: newCol, passLoc: newPassIdx, destIdx: destIdx)
        return [(1.0, newState, reward, terminated)]
    }
    
    public func actionMask(for state: Int) -> [Int] {
        var mask = [Int](repeating: 0, count: 6)
        let (taxiRow, taxiCol, passLoc, destIdx) = Self.decode(state)
        
        if taxiRow < 4 { mask[0] = 1 }
        if taxiRow > 0 { mask[1] = 1 }
        if taxiCol < 4 && canMoveEast(row: taxiRow, col: taxiCol) { mask[2] = 1 }
        if taxiCol > 0 && canMoveWest(row: taxiRow, col: taxiCol) { mask[3] = 1 }
        
        if passLoc < 4 && (taxiRow, taxiCol) == Self.locs[passLoc] {
            mask[4] = 1
        }
        
        if passLoc == 4 {
            let taxiLoc = (taxiRow, taxiCol)
            if taxiLoc == Self.locs[destIdx] || Self.locs.contains(where: { $0 == taxiLoc }) {
                mask[5] = 1
            }
        }
        
        return mask
    }
    
    private func prepareKey(with seed: UInt64?) -> MLXArray {
        if let seed {
            _key = MLX.key(seed)
        } else if _key == nil {
            _key = MLX.key(UInt64.random(in: 0...UInt64.max))
        }
        return _key!
    }
    
    public func reset(
        seed: UInt64? = nil,
        options: [String: Any]? = nil
    ) -> ResetResult {
        let key = prepareKey(with: seed)
        let (sampleKey, nextKey) = MLX.split(key: key)
        _key = nextKey
        
        let epsilon = MLXArray(1e-9, dtype: .float32)
        let probs = MLXArray(initialStateDistrib.map { Float($0) })
        let logits = MLX.log(probs + epsilon)
        let sampledState = MLX.categorical(logits, key: sampleKey)
        s = Int(sampledState.item(Int32.self))
        
        lastAction = nil
        taxiOrientation = 0
        
        if ficklePassenger {
            let (fickleKey, nk) = MLX.split(key: _key!)
            _key = nk
            let fickleRand = MLX.uniform(0..<1, key: fickleKey).item(Float.self)
            fickleStep = fickleRand < 0.3
        } else {
            fickleStep = false
        }
        
        return (obs: s, info: ["prob": 1.0, "action_mask": actionMask(for: s)])
    }
    
    public func step(_ action: Action) -> StepResult {
        guard let transitions = P[s][action], !transitions.isEmpty else {
            fatalError("Invalid state or action")
        }
        
        let (sampleKey, nextKey) = MLX.split(key: _key!)
        _key = nextKey
        
        let probs = transitions.map { Float($0.prob) }
        let epsilon = MLXArray(1e-9, dtype: .float32)
        let logits = MLX.log(MLXArray(probs) + epsilon)
        let i = Int(MLX.categorical(logits, key: sampleKey).item(Int32.self))
        
        let (p, newState, reward, terminated) = transitions[i]
        
        let (shadowRow, shadowCol, shadowPassLoc, shadowDestIdx) = Self.decode(s)
        var (taxiRow, taxiCol, passLoc, destIdx) = Self.decode(newState)
        
        if ficklePassenger && fickleStep && shadowPassLoc == 4 &&
            (taxiRow != shadowRow || taxiCol != shadowCol) {
            fickleStep = false
            
            var possibleDestinations = [Int]()
            for i in 0..<Self.locs.count {
                if i != shadowDestIdx {
                    possibleDestinations.append(i)
                }
            }
            
            let (destKey, nk) = MLX.split(key: _key!)
            _key = nk
            let destRand = MLX.randInt(low: 0, high: possibleDestinations.count, key: destKey)
            destIdx = possibleDestinations[Int(destRand.item(Int32.self))]
            
            s = Self.encode(taxiRow: taxiRow, taxiCol: taxiCol, passLoc: passLoc, destIdx: destIdx)
        } else {
            s = newState
        }
        
        lastAction = action
        if action <= 3 {
            taxiOrientation = action
        }
        
        return (
            obs: s,
            reward: reward,
            terminated: terminated,
            truncated: false,
            info: ["prob": p, "action_mask": actionMask(for: s)]
        )
    }
    
    public func renderAnsi() -> String {
        _renderText()
    }
    
    @discardableResult
    public func render() -> Any? {
        guard let mode = render_mode else {
            if let specId = spec?.id {
                print("[Gymnazo] render() called without render_mode. Set render_mode when creating \(specId).")
            }
            return nil
        }
        
        switch mode {
        case "ansi":
            return _renderText()
        case "human", "rgb_array":
#if canImport(SwiftUI)
            let snapshot = currentSnapshot
            let result = DispatchQueue.main.sync {
                Taxi.renderGUI(snapshot: snapshot, mode: mode)
            }
            if mode == "rgb_array" {
                lastRGBFrame = result as! CGImage?
            }
            return result
#else
            return nil
#endif
        default:
            print("[Gymnazo] Unsupported render_mode \(mode).")
            return nil
        }
    }
    
    private func _renderText() -> String {
        var out = desc.map { $0.map { Character(UnicodeScalar($0)) } }
        let (taxiRow, taxiCol, passIdx, destIdx) = Self.decode(s)
        
        let taxiCharIndex = 2 * taxiCol + 1
        let taxiRowIndex = 1 + taxiRow
        
        if passIdx < 4 {
            out[taxiRowIndex][taxiCharIndex] = "T"
            let (pi, pj) = Self.locs[passIdx]
            out[1 + pi][2 * pj + 1] = "P"
        } else {
            out[taxiRowIndex][taxiCharIndex] = "@"
        }
        
        let (di, dj) = Self.locs[destIdx]
        if out[1 + di][2 * dj + 1] != "T" && out[1 + di][2 * dj + 1] != "@" && out[1 + di][2 * dj + 1] != "P" {
            out[1 + di][2 * dj + 1] = "D"
        }
        
        var result = out.map { String($0) }.joined(separator: "\n")
        
        if let action = lastAction {
            let actionNames = ["South", "North", "East", "West", "Pickup", "Dropoff"]
            result += "\n (\(actionNames[action]))"
        }
        
        result += "\nLegend: T=Taxi, @=Taxi+Passenger, P=Passenger, D=Destination"
        
        return result
    }
    
#if canImport(SwiftUI)
    @MainActor
    private static func renderGUI(snapshot: TaxiRenderSnapshot, mode: String) -> Any? {
        let view = TaxiCanvasView(snapshot: snapshot)
        
        switch mode {
        case "human":
#if canImport(PlaygroundSupport)
            PlaygroundPage.current.setLiveView(view)
#else
            print("[Gymnazo] SwiftUI Canvas available via TaxiCanvasView; integrate it into your app UI.")
#endif
            return nil
        case "rgb_array":
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
        default:
            return nil
        }
    }
    
    public var latestRGBFrame: CGImage? {
        lastRGBFrame
    }
    
    public var currentSnapshot: TaxiRenderSnapshot {
        let (taxiRow, taxiCol, passIdx, destIdx) = Self.decode(s)
        return TaxiRenderSnapshot(
            taxiRow: taxiRow,
            taxiCol: taxiCol,
            passIdx: passIdx,
            destIdx: destIdx,
            taxiOrientation: taxiOrientation,
            lastAction: lastAction
        )
    }
    
    @MainActor
    public func humanView() -> TaxiCanvasView {
        TaxiCanvasView(snapshot: currentSnapshot)
    }
#endif
}

#if canImport(SwiftUI)
/// Taxi SwiftUI Snapshot
public struct TaxiRenderSnapshot: Sendable, Equatable {
    public let taxiRow: Int
    public let taxiCol: Int
    public let passIdx: Int
    public let destIdx: Int
    public let taxiOrientation: Int
    public let lastAction: Int?
}

/// SwiftUI Canvas view
public struct TaxiCanvasView: View {
    public let snapshot: TaxiRenderSnapshot
    
    public init(snapshot: TaxiRenderSnapshot) {
        self.snapshot = snapshot
    }
    
    private let cellSize: CGFloat = 60
    private let gridRows: Int = 5
    private let gridCols: Int = 5
    
    private let locColors: [Color] = [.red, .green, .yellow, .blue]
    private let locLabels: [String] = ["R", "G", "Y", "B"]
    private let locs: [(row: Int, col: Int)] = [(0, 0), (0, 4), (4, 0), (4, 3)]
    
    private let walls: [((Int, Int), (Int, Int))] = [
        ((0, 1), (1, 1)),
        ((3, 0), (4, 0)),
        ((3, 2), (4, 2)),
    ]
    
    public var body: some View {
        Canvas { context, size in
            drawBackground(context: &context, size: size)
            drawGrid(context: &context, size: size)
            drawLocations(context: &context)
            drawWalls(context: &context)
            drawDestination(context: &context)
            drawPassenger(context: &context)
            drawTaxi(context: &context)
        }
        .frame(width: CGFloat(gridCols) * cellSize + 40, height: CGFloat(gridRows) * cellSize + 60)
    }
    
    private func cellOrigin(row: Int, col: Int) -> CGPoint {
        CGPoint(x: 20 + CGFloat(col) * cellSize, y: 20 + CGFloat(row) * cellSize)
    }
    
    private func cellCenter(row: Int, col: Int) -> CGPoint {
        let origin = cellOrigin(row: row, col: col)
        return CGPoint(x: origin.x + cellSize / 2, y: origin.y + cellSize / 2)
    }
    
    private func drawBackground(context: inout GraphicsContext, size: CGSize) {
        let bgColor = Color(red: 0.2, green: 0.2, blue: 0.25)
        context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(bgColor))
    }
    
    private func drawGrid(context: inout GraphicsContext, size: CGSize) {
        let gridColor = Color.gray.opacity(0.3)
        let stroke = StrokeStyle(lineWidth: 1)
        
        for row in 0..<gridRows {
            for col in 0..<gridCols {
                let origin = cellOrigin(row: row, col: col)
                let rect = CGRect(origin: origin, size: CGSize(width: cellSize, height: cellSize))
                
                context.fill(Path(rect), with: .color(Color(red: 0.3, green: 0.3, blue: 0.35)))
                context.stroke(Path(rect), with: .color(gridColor), style: stroke)
            }
        }
    }
    
    private func drawLocations(context: inout GraphicsContext) {
        for (i, loc) in locs.enumerated() {
            let origin = cellOrigin(row: loc.row, col: loc.col)
            let rect = CGRect(origin: origin, size: CGSize(width: cellSize, height: cellSize))
            
            context.fill(Path(rect), with: .color(locColors[i].opacity(0.4)))
            
            context.draw(
                Text(locLabels[i])
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(.white),
                at: CGPoint(x: origin.x + 12, y: origin.y + 12)
            )
        }
    }
    
    private func drawWalls(context: inout GraphicsContext) {
        let wallColor = Color.white
        let wallWidth: CGFloat = 4
        
        for (start, end) in walls {
            let startOrigin = cellOrigin(row: start.0, col: start.1)
            let endOrigin = cellOrigin(row: end.0, col: end.1)
            
            var path = Path()
            path.move(to: CGPoint(x: startOrigin.x, y: startOrigin.y))
            path.addLine(to: CGPoint(x: endOrigin.x, y: endOrigin.y + cellSize))
            
            context.stroke(path, with: .color(wallColor), style: StrokeStyle(lineWidth: wallWidth, lineCap: .round))
        }
    }
    
    private func drawDestination(context: inout GraphicsContext) {
        let destLoc = locs[snapshot.destIdx]
        let center = cellCenter(row: destLoc.row, col: destLoc.col)
        
        let flagSize: CGFloat = 20
        let poleHeight: CGFloat = 30
        
        var polePath = Path()
        polePath.move(to: CGPoint(x: center.x + 10, y: center.y + 15))
        polePath.addLine(to: CGPoint(x: center.x + 10, y: center.y + 15 - poleHeight))
        context.stroke(polePath, with: .color(.white), style: StrokeStyle(lineWidth: 2))
        
        var flagPath = Path()
        flagPath.move(to: CGPoint(x: center.x + 10, y: center.y + 15 - poleHeight))
        flagPath.addLine(to: CGPoint(x: center.x + 10 + flagSize, y: center.y + 15 - poleHeight + 8))
        flagPath.addLine(to: CGPoint(x: center.x + 10, y: center.y + 15 - poleHeight + 16))
        flagPath.closeSubpath()
        context.fill(flagPath, with: .color(Color(red: 1.0, green: 0.0, blue: 1.0)))
    }
    
    private func drawPassenger(context: inout GraphicsContext) {
        if snapshot.passIdx < 4 {
            let passLoc = locs[snapshot.passIdx]
            let center = cellCenter(row: passLoc.row, col: passLoc.col)
            drawPersonShape(context: &context, at: CGPoint(x: center.x - 12, y: center.y), scale: 1.0, color: .cyan)
        }
    }
    
    private func drawPersonShape(context: inout GraphicsContext, at point: CGPoint, scale: CGFloat, color: Color) {
        let headRadius: CGFloat = 6 * scale
        let headCenter = CGPoint(x: point.x, y: point.y - 10 * scale)
        let headPath = Path(ellipseIn: CGRect(
            x: headCenter.x - headRadius,
            y: headCenter.y - headRadius,
            width: headRadius * 2,
            height: headRadius * 2
        ))
        context.fill(headPath, with: .color(color))
        
        var bodyPath = Path()
        bodyPath.move(to: CGPoint(x: point.x, y: point.y - 4 * scale))
        bodyPath.addLine(to: CGPoint(x: point.x, y: point.y + 10 * scale))
        context.stroke(bodyPath, with: .color(color), style: StrokeStyle(lineWidth: 2 * scale, lineCap: .round))
        
        var armsPath = Path()
        armsPath.move(to: CGPoint(x: point.x - 8 * scale, y: point.y + 2 * scale))
        armsPath.addLine(to: CGPoint(x: point.x + 8 * scale, y: point.y + 2 * scale))
        context.stroke(armsPath, with: .color(color), style: StrokeStyle(lineWidth: 2 * scale, lineCap: .round))
        
        var legsPath = Path()
        legsPath.move(to: CGPoint(x: point.x, y: point.y + 10 * scale))
        legsPath.addLine(to: CGPoint(x: point.x - 6 * scale, y: point.y + 20 * scale))
        legsPath.move(to: CGPoint(x: point.x, y: point.y + 10 * scale))
        legsPath.addLine(to: CGPoint(x: point.x + 6 * scale, y: point.y + 20 * scale))
        context.stroke(legsPath, with: .color(color), style: StrokeStyle(lineWidth: 2 * scale, lineCap: .round))
    }
    
    private func drawTaxi(context: inout GraphicsContext) {
        let center = cellCenter(row: snapshot.taxiRow, col: snapshot.taxiCol)
        let hasPassenger = snapshot.passIdx == 4
        
        let bodyColor = hasPassenger ? Color.green : Color.yellow
        let bodyWidth: CGFloat = 36
        let bodyHeight: CGFloat = 20
        let roofWidth: CGFloat = 20
        let roofHeight: CGFloat = 12
        let wheelRadius: CGFloat = 5
        
        let bodyRect = CGRect(
            x: center.x - bodyWidth / 2,
            y: center.y - bodyHeight / 2 + 4,
            width: bodyWidth,
            height: bodyHeight
        )
        let bodyPath = Path(roundedRect: bodyRect, cornerRadius: 4)
        context.fill(bodyPath, with: .color(bodyColor))
        context.stroke(bodyPath, with: .color(.black), style: StrokeStyle(lineWidth: 1))
        
        let roofRect = CGRect(
            x: center.x - roofWidth / 2,
            y: center.y - bodyHeight / 2 - roofHeight + 6,
            width: roofWidth,
            height: roofHeight
        )
        let roofPath = Path(roundedRect: roofRect, cornerRadius: 3)
        context.fill(roofPath, with: .color(bodyColor.opacity(0.9)))
        context.stroke(roofPath, with: .color(.black), style: StrokeStyle(lineWidth: 1))
        
        let windowRect = CGRect(
            x: center.x - roofWidth / 2 + 3,
            y: center.y - bodyHeight / 2 - roofHeight + 9,
            width: roofWidth - 6,
            height: roofHeight - 6
        )
        context.fill(Path(roundedRect: windowRect, cornerRadius: 2), with: .color(Color(white: 0.8)))
        
        let leftWheelCenter = CGPoint(x: center.x - bodyWidth / 3, y: center.y + bodyHeight / 2 + 2)
        let rightWheelCenter = CGPoint(x: center.x + bodyWidth / 3, y: center.y + bodyHeight / 2 + 2)
        
        let leftWheelPath = Path(ellipseIn: CGRect(
            x: leftWheelCenter.x - wheelRadius,
            y: leftWheelCenter.y - wheelRadius,
            width: wheelRadius * 2,
            height: wheelRadius * 2
        ))
        let rightWheelPath = Path(ellipseIn: CGRect(
            x: rightWheelCenter.x - wheelRadius,
            y: rightWheelCenter.y - wheelRadius,
            width: wheelRadius * 2,
            height: wheelRadius * 2
        ))
        
        context.fill(leftWheelPath, with: .color(.black))
        context.fill(rightWheelPath, with: .color(.black))
        
        let hubRadius: CGFloat = 2
        context.fill(Path(ellipseIn: CGRect(
            x: leftWheelCenter.x - hubRadius,
            y: leftWheelCenter.y - hubRadius,
            width: hubRadius * 2,
            height: hubRadius * 2
        )), with: .color(.gray))
        context.fill(Path(ellipseIn: CGRect(
            x: rightWheelCenter.x - hubRadius,
            y: rightWheelCenter.y - hubRadius,
            width: hubRadius * 2,
            height: hubRadius * 2
        )), with: .color(.gray))
        
        if hasPassenger {
            drawPersonShape(
                context: &context,
                at: CGPoint(x: center.x + 18, y: center.y - 12),
                scale: 0.5,
                color: .cyan
            )
        }
    }
}
#endif

