import Foundation
import MLX
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

/// Frozen lake involves crossing a frozen lake from start to goal without falling into any holes
/// by walking over the frozen lake.
///
/// The player may not always move in the intended direction due to the slippery nature of the frozen lake.
///
/// ## Description
///
/// The game starts with the player at location `[0,0]` of the frozen lake grid world with the
/// goal located at the far extent of the world (e.g., `[3,3]` for the 4x4 environment).
///
/// Holes in the ice are distributed in set locations when using a pre-determined map
/// or in random locations when a random map is generated.
/// Randomly generated worlds will always have a path to the goal.
///
/// The player makes moves until they reach the goal or fall in a hole.
///
/// The lake is slippery (unless disabled) so the player may move perpendicular
/// to the intended direction sometimes (see `isSlippery` parameter).
///
/// ## Action Space
///
/// The action is an `Int` in the range `{0, 3}` indicating which direction to move the player:
///
/// | Action | Direction |
/// |--------|-----------|
/// | 0      | Left      |
/// | 1      | Down      |
/// | 2      | Right     |
/// | 3      | Up        |
///
/// ## Observation Space
///
/// The observation is an `Int` representing the player's current position as:
/// ```
/// current_row * ncols + current_col
/// ```
/// where both row and col start at 0.
///
/// For example, the goal position in the 4x4 map can be calculated as: `3 * 4 + 3 = 15`.
///
/// ## Starting State
///
/// The episode starts with the player in state `0` (location `[0, 0]`).
///
/// ## Rewards
///
/// Default reward schedule:
/// - Reach goal: **+1**
/// - Reach hole: **0**
/// - Reach frozen: **0**
///
/// ## Episode End
///
/// The episode ends if the following happens:
///
/// **Termination:**
/// 1. The player moves into a hole.
/// 2. The player reaches the goal at `max(nrow) * max(ncol) - 1`.
///
/// **Truncation:**
/// - When wrapped with ``TimeLimit``, the episode truncates after 100 steps (4x4) or 200 steps (8x8).
///
/// ## Information
///
/// `step()` and `reset()` return info with the following keys:
/// - `prob`: Transition probability for the state (affected by `isSlippery`).
///
/// ## Arguments
///
/// - `render_mode`: The render mode (`"human"`, `"ansi"`, or `"rgb_array"`).
/// - `desc`: Custom map as an array of strings (e.g., `["SFFF", "FHFH", "FFFH", "HFFG"]`).
/// - `map_name`: Predefined map name (`"4x4"` or `"8x8"`). Ignored if `desc` is provided.
/// - `isSlippery`: If `true`, the player moves in the intended direction with probability `successRate`,
///   otherwise moves perpendicular with equal probability.
/// - `successRate`: Probability of moving in the intended direction when `isSlippery` is `true` (default: 1/3).
///
/// ## Map Tiles
///
/// | Tile | Description |
/// |------|-------------|
/// | S    | Start       |
/// | F    | Frozen      |
/// | H    | Hole        |
/// | G    | Goal        |
///
/// ## Version History
///
/// - v1: Initial Swift port
public final class FrozenLake: Env {
    private enum Direction: Int, CaseIterable {
        case left = 0
        case down = 1
        case right = 2
        case up = 3
    }
    
    private func toInt(_ action: MLXArray) -> Int {
        Int(action.item(Int32.self))
    }
    
    private func toMLX(_ state: Int) -> MLXArray {
        MLXArray([Int32(state)])
    }

    public static let MAPS: [String : [String]] = [
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

    /// Checks if a valid path exists from start (0,0) to goal using depth-first search.
    /// - Parameters:
    ///   - board: 2D array representing the map tiles
    ///   - maxSize: Size of the square grid
    /// - Returns: `true` if a path from start to goal exists
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

    /// Generates a random valid map (one that has a path from start to goal).
    ///
    /// The map is generated by sampling frozen vs hole tiles with probability `p`,
    /// then validating connectivity using depth-first search. Generation repeats
    /// until a solvable map is produced.
    ///
    /// ```swift
    /// let randomMap = FrozenLake.generateRandomMap(size: 8, p: 0.8, seed: 42)
    /// let env = FrozenLake(desc: randomMap)
    /// ```
    ///
    /// - Parameters:
    ///   - size: Size of each side of the grid (default: 8)
    ///   - p: Probability that a tile is frozen (default: 0.8)
    ///   - seed: Optional seed for reproducible map generation
    /// - Returns: Array of strings representing a valid map
    public static func generateRandomMap(
        size: Int = 8,
        p: Float = 0.8,
        seed: Int? = nil
    ) -> [String] {
        var valid: Bool = false
        var board: [[String]] = []

        var master: MLXArray

        if let seed = seed {
            master = MLX.key(UInt64(seed))
        } else {
            master = MLX.key(UInt64.random(in: 0...UInt64.max))
        }

        while !valid {
            let loopKey: (MLXArray, MLXArray) = MLX.split(key: master)
            master = loopKey.1

            let uniform: MLXArray = MLX.uniform(0 ..< 1, [size, size], key: loopKey.0)

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

    public let actionSpace: any Space
    public let observationSpace: any Space
    public var spec: EnvSpec?
    public var renderMode: RenderMode?
    
    private let descMatrix: [[Character]]
    private let nrow: Int
    private let ncol: Int

#if canImport(SwiftUI)
    private var lastRGBFrame: CGImage?
#endif

    public typealias Transition = (prob: Double, nextState: Int, reward: Double, terminated: Bool)
    private let P: [[[Transition]]]

    private var s: Int
    private var lastAction: Int?
    
    private var _key: MLXArray?
    private var _seed: UInt64?

    private let initial_state_logits: MLXArray

    private func prepareKey(with seed: UInt64?) throws -> MLXArray {
        if let seed {
            self._seed = seed
            self._key = MLX.key(seed)
        } else if self._key == nil {
            let randomSeed = UInt64.random(in: 0...UInt64.max)
            self._seed = randomSeed
            self._key = MLX.key(randomSeed)
        }

        guard let key = self._key else {
            throw GymnazoError.invalidState("Failed to initialize RNG key")
        }

        return key
    }

    public init(
        renderMode: RenderMode? = nil,
        desc: [String]? = nil,
        map_name: String = "4x4",
        isSlippery: Bool = true,
        successRate: Float = (1.0 / 3.0)
    ) {
        self.renderMode = renderMode
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
        self.observationSpace = Discrete(n: nS)
        self.actionSpace = Discrete(n: nA)

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
            
            var reward = reward_schedule.2
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
                        continue
                    }
                    var li: [Transition] = []
                    let letter = mapChars[r][c]
                    
                    if "GH".contains(letter) {
                        li.append((prob: 1.0, nextState: s, reward: 0.0, terminated: true))
                    } else {
                        if isSlippery {
                            for b_int in [(a_int - 1 + 4) % 4, a_int, (a_int + 1) % 4] {
                                guard let b = FrozenLake.Direction(rawValue: b_int) else {
                                    continue
                                }
                                let p = (b == a) ? successProb : failProb
                                let (s_new, r_new, t_new) = update_prob_matrix(r, c, b)
                                li.append((prob: p, nextState: s_new, reward: r_new, terminated: t_new))
                            }
                        } else {
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

    public convenience init(
        renderMode: RenderMode? = nil,
        desc: [String]? = nil,
        map_name: String = "4x4",
        isSlippery: Bool = true
    ) {
        self.init(
            renderMode: renderMode,
            desc: desc,
            map_name: map_name,
            isSlippery: isSlippery,
            successRate: (1.0 / 3.0)
        )
    }

    public func reset(
        seed: UInt64? = nil,
        options: EnvOptions? = nil
    ) throws -> Reset {
        
        let key = try self.prepareKey(with: seed)
        
        let (resetKey, nextKey) = MLX.split(key: key)
        self._key = nextKey
        
        let s_ml: MLXArray = MLX.categorical(self.initial_state_logits, key: resetKey)
        self.s = s_ml.item(Int.self)
        self.lastAction = nil
        
        return Reset(obs: toMLX(self.s), info: ["prob": 1.0])
    }

    public func step(_ action: MLXArray) throws -> Step {
        let a = toInt(action)
        guard actionSpace.contains(action) else {
            throw GymnazoError.invalidAction("Invalid action: \(a)")
        }
        let transitions = self.P[self.s][a]
        
        guard let key = self._key else {
            throw GymnazoError.invalidState("Env must be seeded with reset(seed:)")
        }
        let (stepKey, nextKey) = MLX.split(key: key)
        self._key = nextKey
        
        let probs = transitions.map { Float($0.prob) }
        let epsilon = MLXArray(1e-9, dtype: .float32)
        let prob_logits = MLX.log(MLXArray(probs) + epsilon)
        
        let i = MLX.categorical(prob_logits, key: stepKey).item(Int.self)
        
        let (p, s, r, t) = transitions[i]
        
        self.s = s
        self.lastAction = a
        
        return Step(obs: toMLX(self.s), reward: r, terminated: t, truncated: false, info: ["prob": .double(p)])
    }

    /// returns an ansi representation of the current grid, independent of `render_mode`.
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
    private static func _renderGUI(snapshot: FrozenLakeRenderSnapshot, mode: RenderMode) -> CGImage? {
#if canImport(SwiftUI)
        let view = FrozenLakeCanvasView(snapshot: snapshot)

        switch mode {
        case .human:
#if canImport(PlaygroundSupport)
            PlaygroundPage.current.setLiveView(view)
#else
            print("[Gymnazo] SwiftUI Canvas available via FrozenLakeCanvasView; integrate it into your app UI.")
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
        case .ansi:
            return nil
        case .statePixels:
            return nil
        }
#else
        print("[Gymnazo] SwiftUI is not available; falling back to ANSI render.")
        return nil
#endif
    }

#if canImport(SwiftUI)
    /// last rendered rgb frame when using the "rgb_array" render mode.
    public var latestRGBFrame: CGImage? {
        lastRGBFrame
    }

    @MainActor
    public func renderRGBArray() -> CGImage? {
        let image = Self._renderGUI(snapshot: currentSnapshot, mode: .rgbArray)
        lastRGBFrame = image
        return image
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
    public let rows: Int
    public let cols: Int
    public let tiles: [[Character]]
    public let playerIndex: Int
    public let lastAction: Int?

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
