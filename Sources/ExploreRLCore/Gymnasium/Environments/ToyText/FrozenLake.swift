//
// FrozenLake.swift
//

import Foundation
import MLXRandom
import MLX
import Playgrounds
import CoreGraphics
#if canImport(SwiftUI)
import SwiftUI
#if os(macOS)
import AppKit
#elseif os(iOS) || os(tvOS) || os(visionOS)
import UIKit
#endif
#endif
#if canImport(PlaygroundSupport)
import PlaygroundSupport
#endif

public final class FrozenLake: Environment {
    private enum Direction: Int, CaseIterable {
        case left = 0
        case down = 1
        case right = 2
        case up = 3
    }

    private static let MAPS: [String : [String]] = [
        "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
        "8x8": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ],
    ]

    /// DFS to check if the map is valid
    private static func isValid(board: [[String]], maxSize: Int) -> Bool {
        var frontier: [Coord] = [Coord(row: 0, col: 0)]
        var discovered: Set<Coord> = []
        while !frontier.isEmpty {
            let coord: Coord? = frontier.popLast()
            if let coord = coord, !discovered.contains(coord) {
                discovered.insert(coord)
                let directions: [Coord] = [
                    Coord(row: 1, col: 0),
                    Coord(row: 0, col: 1),
                    Coord(row: -1, col: 0),
                    Coord(row: 0, col: -1),
                ]
                for direction in directions {
                    let newRow = coord.row + direction.row
                    let newCol = coord.col + direction.col

                    if newRow < 0 || newRow >= maxSize || newCol < 0 || newCol >= maxSize {
                        continue
                    }
                    if board[newRow][newCol] == "G" {
                        return true
                    }
                    if board[newRow][newCol] != "H" {
                        frontier.append(Coord(row: newRow, col: newCol))
                    }
                }
            }
        }
        return false
    }

    /// generate a random frozen lake map like gymnasium's generate_random_map
    /// by sampling frozen vs hole tiles with probability `p` and checking connectivity.
    public static func generateRandomMap(
        size: Int = 8,
        p: Float = 0.8,
        seed: Int? = nil
    ) -> [String] {
        var valid: Bool = false
        var board: [[String]] = []

        var master: MLXArray

        if let seed = seed {
            master = MLXRandom.key(UInt64(seed))
        } else {
            master = MLXRandom.key(UInt64.random(in: 0...UInt64.max))
        }

        while !valid {
            let loopKey: (MLXArray, MLXArray) = MLXRandom.split(key: master)
            master = loopKey.1

            let uniform: MLXArray = MLXRandom.uniform(0 ..< 1, [size, size], key: loopKey.0)

            let boardML: MLXArray = MLX.which(uniform .< p, MLXArray(0), MLXArray(1))
            eval(boardML)
            let boardData = boardML.asArray(Int32.self)

            board = []
            for i in 0..<size {
                var row: [String] = []
                for j in 0..<size {
                    let val = boardData[i * size + j]
                    row.append(val == 0 ? "F" : "H")
                }
                board.append(row)
            }

            board[0][0] = "S"
            board[size - 1][size - 1] = "G"
            valid = Self.isValid(board: board, maxSize: size)

        }
        return board.map { $0.joined() }
    }

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
    
    private let descMatrix: [[Character]]
    private let nrow: Int
    private let ncol: Int

#if canImport(SwiftUI)
    private var lastRGBFrame: CGImage?
#endif

    public typealias Transition = (prob: Double, nextState: Int, reward: Double, terminated: Bool)
    private let P: [[[Transition]]]

    private var s: Int
    private var lastAction: Action?
    
    private var _key: MLXArray?
    private var _seed: UInt64?

    private let initial_state_logits: MLXArray

    private func prepareKey(with seed: UInt64?) -> MLXArray {
        if let seed {
            self._seed = seed
            self._key = MLXRandom.key(seed)
        } else if self._key == nil {
            let randomSeed = UInt64.random(in: 0...UInt64.max)
            self._seed = randomSeed
            self._key = MLXRandom.key(randomSeed)
        }

        guard let key = self._key else {
            fatalError("Failed to initialize RNG key")
        }

        return key
    }

    public init(
        render_mode: String? = nil,
        desc: [String]? = nil,
        map_name: String = "4x4",
        isSlippery: Bool = true,
        successRate: Float = (1.0 / 3.0)
    ) {
        self.render_mode = render_mode
        let reward_schedule: (Double, Double, Double) = (1.0, 0.0, 0.0)

        var mapDesc: [String]
        if let desc: [String] = desc {
            mapDesc = desc
        } else {
            mapDesc = Self.MAPS[map_name]!
        }

        let mapChars: [[String.Element]] = mapDesc.map { Array($0) }
        let nrow = mapChars.count
        let ncol = mapChars[0].count
        self.nrow = nrow
        self.ncol = ncol
        self.descMatrix = mapChars

        let nA: Int = 4
        let nS: Int = nrow * ncol
        self.observation_space = Discrete(n: nS)
        self.action_space = Discrete(n: nA)

        var startStateLogits: [Float] = [Float](repeating: -Float.infinity, count: nS)
        var s_count: Int = 0
        for r: Int in 0..<nrow {
            for c: Int in 0..<ncol {
                if mapChars[r][c] == "S" {
                    startStateLogits[r * ncol + c] = 0.0
                    s_count += 1
                }
            }
        }

        if s_count > 1 {
            startStateLogits = startStateLogits.map { $0 == 0.0 ? 0.0 : -Float.infinity }
        }
        self.initial_state_logits = MLXArray(startStateLogits)
        let successProb: Double = Double(successRate)
        let failProb: Double = (1.0 - successProb) / 2.0
        
        func to_s(_ r: Int, _ c: Int) -> Int { r * ncol + c }
        
        func inc(_ r: Int, _ c: Int, _ a: FrozenLake.Direction) -> (Int, Int) {
            var newRow = r, newCol = c
            switch a {
            case .left: newCol = max(c - 1, 0)
            case .down: newRow = min(r + 1, nrow - 1)
            case .right: newCol = min(c + 1, ncol - 1)
            case .up: newRow = max(r - 1, 0)
            }
            return (newRow, newCol)
        }

        func update_prob_matrix(_ r: Int, _ c: Int, _ a: FrozenLake.Direction) -> (Int, Double, Bool) {
            let (newRow, newCol) = inc(r, c, a)
            let newState = to_s(newRow, newCol)
            let newLetter = mapChars[newRow][newCol]
            let terminated = "GH".contains(newLetter)
            
            var reward = reward_schedule.2 // Frozen
            if newLetter == "G" { reward = reward_schedule.0 }
            else if newLetter == "H" { reward = reward_schedule.1 }
            
            return (newState, reward, terminated)
        }

        var P_temp: [[[Transition]]] = Array(
            repeating: Array(repeating: [], count: nA),
            count: nS
        )
        
        for r in 0..<nrow {
            for c in 0..<ncol {
                let s = to_s(r, c)
                for a_int in 0..<nA {
                    guard let a = FrozenLake.Direction(rawValue: a_int) else {
                        fatalError("Invalid action index \(a_int)")
                    }
                    var li: [Transition] = []
                    let letter = mapChars[r][c]
                    
                    if "GH".contains(letter) {
                        // Terminal state: 100% prob to stay, 0 reward, terminated
                        li.append((prob: 1.0, nextState: s, reward: 0.0, terminated: true))
                    } else {
                        if isSlippery {
                            for b_int in [(a_int - 1 + 4) % 4, a_int, (a_int + 1) % 4] {
                                guard let b = FrozenLake.Direction(rawValue: b_int) else {
                                    fatalError("Invalid action index \(b_int)")
                                }
                                let p = (b == a) ? successProb : failProb
                                let (s_new, r_new, t_new) = update_prob_matrix(r, c, b)
                                li.append((prob: p, nextState: s_new, reward: r_new, terminated: t_new))
                            }
                        } else {
                            // Not slippery
                            let (s_new, r_new, t_new) = update_prob_matrix(r, c, a)
                            li.append((prob: 1.0, nextState: s_new, reward: r_new, terminated: t_new))
                        }
                    }
                    P_temp[s][a_int] = li
                }
            }
        }
        self.P = P_temp

        self.s = 0
        self.lastAction = nil
        self._key = nil
        self._seed = nil
    }

    public func reset(
        seed: UInt64? = nil,
        options: [String : Any]? = nil
    ) -> ResetResult {
        
        let key = self.prepareKey(with: seed)
        
        // 2. Split key
        let (resetKey, nextKey) = MLXRandom.split(key: key)
        self._key = nextKey
        
        // 3. Sample initial state
        // Python: `self.s = categorical_sample(self.initial_state_distrib, ...)`
        let s_ml: MLXArray = MLXRandom.categorical(self.initial_state_logits, key: resetKey)
        let sampledState: Int32 = s_ml.item() as Int32
        self.s = Int(sampledState)
        self.lastAction = nil
        
        return (obs: self.s, info: ["prob": 1.0])
    }

    public func step(_ action: Action) -> StepResult {
        
        // 1. Get transitions for (s, a)
        let transitions = self.P[self.s][action]
        
        // 2. Get and split key
        guard let key = self._key else {
            fatalError("Env must be seeded with reset(seed:)")
        }
        let (stepKey, nextKey) = MLXRandom.split(key: key)
        self._key = nextKey
        
        // 3. Sample an outcome
        // Python: `i = categorical_sample([t[0] for t in transitions], ...)`
        let probs = transitions.map { Float($0.prob) }
        let epsilon = MLXArray(1e-9, dtype: .float32)
        let prob_logits = MLX.log(MLXArray(probs) + epsilon)
        
        let i_ml = MLXRandom.categorical(prob_logits, key: stepKey)
        let sampledIndex: Int32 = i_ml.item() as Int32
        let i = Int(sampledIndex)
        
        let (p, s, r, t) = transitions[i]
        
        // 4. Update state
        self.s = s
        self.lastAction = action
        
        return (obs: self.s, reward: r, terminated: t, truncated: false, info: ["prob": p])
    }

    /// returns an ansi representation of the current grid, independent of `render_mode`.
    public func renderAnsi() -> String {
        _renderText()
    }

    @discardableResult
    public func render() -> Any? {
        guard let mode = render_mode else {
            if let specId = spec?.id {
                print("[Gymnasium] render() called without render_mode. Set render_mode when creating \(specId).")
            }
            return nil
        }

        switch mode {
        case "ansi":
            return _renderText()
        case "human", "rgb_array":
            #if canImport(SwiftUI)
            let snapshot = self.currentSnapshot
            let result = DispatchQueue.main.sync {
                FrozenLake._renderGUI(snapshot: snapshot, mode: mode)
            }
            if mode == "rgb_array" {
                self.lastRGBFrame = result as! CGImage?
            }
            return result
            #else
            return nil
            #endif
        default:
            print("[Gymnasium] Unsupported render_mode \(mode).")
            return nil
        }
    }

    private func _renderText() -> String {
        var rows: [String] = []
        for r in 0..<nrow {
            var cols: [String] = []
            for c in 0..<ncol {
                let idx = r * ncol + c
                var tile = descMatrix[r][c]
                if idx == s {
                    tile = tile == "H" ? "X" : "P"
                }
                cols.append(String(tile))
            }
            rows.append(cols.joined(separator: " "))
        }
        return rows.joined(separator: "\n")
    }

    @MainActor
    private static func _renderGUI(snapshot: FrozenLakeRenderSnapshot, mode: String) -> Any? {
#if canImport(SwiftUI)
        let view = FrozenLakeCanvasView(snapshot: snapshot)

        switch mode {
        case "human":
#if canImport(PlaygroundSupport)
            PlaygroundPage.current.setLiveView(view)
#else
            print("[Gymnasium] SwiftUI Canvas available via FrozenLakeCanvasView; integrate it into your app UI.")
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
                print("[Gymnasium] rgb_array rendering requires macOS 13/iOS 16.")
                return nil
            }
        default:
            return nil
        }
#else
        print("[Gymnasium] SwiftUI is not available; falling back to ANSI render.")
        return nil
#endif
    }

#if canImport(SwiftUI)
    /// last rendered rgb frame when using the "rgb_array" render mode.
    public var latestRGBFrame: CGImage? {
        lastRGBFrame
    }

    /// snapshot of the current grid state for use with `FrozenLakeCanvasView`.
    public var currentSnapshot: FrozenLakeRenderSnapshot {
        FrozenLakeRenderSnapshot(
            rows: nrow,
            cols: ncol,
            tiles: descMatrix,
            playerIndex: s,
            lastAction: lastAction
        )
    }

    /// convenience for embedding the environment into a swiftui hierarchy.
    @MainActor
    public func humanView() -> FrozenLakeCanvasView {
        FrozenLakeCanvasView(snapshot: currentSnapshot)
    }
#endif

}

#if canImport(SwiftUI)
/// Snapshot of the lake grid used by SwiftUI renderers.
public struct FrozenLakeRenderSnapshot: Sendable {
    let rows: Int
    let cols: Int
    let tiles: [[Character]]
    let playerIndex: Int
    let lastAction: Int?

    func tile(at index: Int) -> Character {
        let row = index / cols
        let col = index % cols
        return tiles[row][col]
    }
}

/// SwiftUI Canvas view that renders a snapshot of FrozenLake.
public struct FrozenLakeCanvasView: View {
    let snapshot: FrozenLakeRenderSnapshot
    /// Creates a canvas view for a given snapshot.
    public init(snapshot: FrozenLakeRenderSnapshot) {
        self.snapshot = snapshot
    }

    public var body: some View {
        GeometryReader { proxy in
            Canvas { context, size in
                drawBackground(context: &context, size: size)
                drawGrid(context: &context, size: size)
                drawPlayer(context: &context, size: size)
            }
        }
        .aspectRatio(CGFloat(snapshot.cols) / CGFloat(snapshot.rows), contentMode: .fit)
    }

    private func cellSize(for size: CGSize) -> CGSize {
        CGSize(width: size.width / CGFloat(snapshot.cols), height: size.height / CGFloat(snapshot.rows))
    }

    private func drawBackground(context: inout GraphicsContext, size: CGSize) {
        let cell = cellSize(for: size)
        for row in 0..<snapshot.rows {
            for col in 0..<snapshot.cols {
                let rect = CGRect(
                    x: CGFloat(col) * cell.width,
                    y: CGFloat(row) * cell.height,
                    width: cell.width,
                    height: cell.height
                )
                let tile = snapshot.tiles[row][col]
                let color: Color
                switch tile {
                case "S": color = Color(red: 0.6, green: 0.8, blue: 1.0)
                case "G": color = Color.green.opacity(0.8)
                case "H": color = Color.blue.opacity(0.5)
                default: color = Color.cyan.opacity(0.7)
                }
                context.fill(Path(rect), with: .color(color))
            }
        }
    }

    private func drawGrid(context: inout GraphicsContext, size: CGSize) {
        let cell = cellSize(for: size)
        let stroke = StrokeStyle(lineWidth: 1)
        for row in 0..<snapshot.rows {
            for col in 0..<snapshot.cols {
                let rect = CGRect(
                    x: CGFloat(col) * cell.width,
                    y: CGFloat(row) * cell.height,
                    width: cell.width,
                    height: cell.height
                )
                context.stroke(Path(rect), with: .color(Color.white.opacity(0.4)), style: stroke)
            }
        }
    }

    private func drawPlayer(context: inout GraphicsContext, size: CGSize) {
        let cell = cellSize(for: size)
        let row = snapshot.playerIndex / snapshot.cols
        let col = snapshot.playerIndex % snapshot.cols
        let rect = CGRect(
            x: CGFloat(col) * cell.width,
            y: CGFloat(row) * cell.height,
            width: cell.width,
            height: cell.height
        )

        var path = Path()
        let insetRect = rect.insetBy(dx: cell.width * 0.2, dy: cell.height * 0.2)
        path.addRoundedRect(in: insetRect, cornerSize: CGSize(width: 6, height: 6))
        let tile = snapshot.tile(at: snapshot.playerIndex)
        let fillColor: Color = tile == "H" ? .red : .orange
        context.fill(path, with: .color(fillColor))

        if let direction = snapshot.lastAction {
            let indicatorSize = CGSize(width: cell.width * 0.2, height: cell.height * 0.2)
            var indicatorRect = CGRect(
                x: insetRect.midX - indicatorSize.width / 2,
                y: insetRect.midY - indicatorSize.height / 2,
                width: indicatorSize.width,
                height: indicatorSize.height
            )
            let offset: CGPoint
            switch direction {
            case 0: offset = CGPoint(x: -indicatorSize.width, y: 0)
            case 1: offset = CGPoint(x: 0, y: indicatorSize.height)
            case 2: offset = CGPoint(x: indicatorSize.width, y: 0)
            case 3: offset = CGPoint(x: 0, y: -indicatorSize.height)
            default: offset = .zero
            }
            indicatorRect.origin.x += offset.x
            indicatorRect.origin.y += offset.y
            context.fill(Path(roundedRect: indicatorRect, cornerRadius: 4), with: .color(.white.opacity(0.9)))
        }
    }
}
#endif
