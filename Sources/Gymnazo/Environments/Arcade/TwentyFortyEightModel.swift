import Foundation

struct TwentyFortyEightModel: Sendable {
    static let boardSize = 4
    static let cellCount = boardSize * boardSize
    static let observationCount = cellCount

    private(set) var tiles = [Int](repeating: 0, count: Self.cellCount)
    private(set) var score = 0

    private var generator: SeededGenerator?
    private var needsReset = true

    mutating func reset(seed: UInt64?) -> [Float] {
        if let seed {
            generator = SeededGenerator(seed: seed)
        } else if generator == nil {
            generator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
        }

        tiles = [Int](repeating: 0, count: Self.cellCount)
        score = 0
        addRandomTile()
        addRandomTile()
        needsReset = false
        return observation
    }

    mutating func step(action: Int) throws -> Result {
        guard !needsReset else {
            throw GymnazoError.stepBeforeReset
        }

        guard let direction = Direction(rawValue: action) else {
            throw GymnazoError.invalidAction("2048 actions must be 0, 1, 2, or 3.")
        }

        let before = tiles
        let mergeScore = move(direction: direction)
        if tiles != before {
            addRandomTile()
        }

        let terminated = !canMove
        needsReset = terminated
        score += mergeScore

        return Result(
            observation: observation,
            reward: Double(mergeScore) + (terminated ? -1 : 0),
            terminated: terminated,
            score: score,
            maxTile: tiles.max() ?? 0
        )
    }

    var snapshot: TwentyFortyEightSnapshot {
        TwentyFortyEightSnapshot(
            tiles: tiles,
            score: score,
            maxTile: tiles.max() ?? 0,
            isGameOver: !canMove
        )
    }

    private var observation: [Float] {
        tiles.map { tile in
            guard tile > 0 else { return 0 }
            return min(Float(log2(Double(tile)) / 16), 1)
        }
    }

    private var canMove: Bool {
        if tiles.contains(0) { return true }

        for y in 0..<Self.boardSize {
            for x in 0..<Self.boardSize {
                let value = tiles[index(x: x, y: y)]
                if x + 1 < Self.boardSize,
                    value == tiles[index(x: x + 1, y: y)]
                {
                    return true
                }
                if y + 1 < Self.boardSize,
                    value == tiles[index(x: x, y: y + 1)]
                {
                    return true
                }
            }
        }
        return false
    }

    private mutating func move(direction: Direction) -> Int {
        var mergeScore = 0
        for fixed in 0..<Self.boardSize {
            let indices = lineIndices(fixed: fixed, direction: direction)
            let values = indices.map { tiles[$0] }
            let collapsed = collapse(values)
            mergeScore += collapsed.reward

            for (index, value) in zip(indices, collapsed.values) {
                tiles[index] = value
            }
        }
        return mergeScore
    }

    private func lineIndices(fixed: Int, direction: Direction) -> [Int] {
        switch direction {
        case .up:
            return (0..<Self.boardSize).map { index(x: fixed, y: $0) }
        case .right:
            return (stride(from: Self.boardSize - 1, through: 0, by: -1))
                .map { index(x: $0, y: fixed) }
        case .down:
            return (stride(from: Self.boardSize - 1, through: 0, by: -1))
                .map { index(x: fixed, y: $0) }
        case .left:
            return (0..<Self.boardSize).map { index(x: $0, y: fixed) }
        }
    }

    private func index(x: Int, y: Int) -> Int {
        y * Self.boardSize + x
    }

    private func collapse(_ values: [Int]) -> CollapseResult {
        let nonZero = values.filter { $0 != 0 }
        var collapsed: [Int] = []
        collapsed.reserveCapacity(Self.boardSize)
        var reward = 0
        var index = 0

        while index < nonZero.count {
            if index + 1 < nonZero.count, nonZero[index] == nonZero[index + 1] {
                let merged = nonZero[index] * 2
                collapsed.append(merged)
                reward += merged
                index += 2
            } else {
                collapsed.append(nonZero[index])
                index += 1
            }
        }

        collapsed.append(contentsOf: repeatElement(0, count: Self.boardSize - collapsed.count))
        return CollapseResult(values: collapsed, reward: reward)
    }

    private mutating func addRandomTile() {
        let empty = tiles.indices.filter { tiles[$0] == 0 }
        guard !empty.isEmpty, var generator else { return }

        let position = empty[Int.random(in: 0..<empty.count, using: &generator)]
        let value = Int.random(in: 0..<10, using: &generator) == 0 ? 4 : 2
        tiles[position] = value
        self.generator = generator
    }

    private struct CollapseResult: Sendable {
        let values: [Int]
        let reward: Int
    }

    struct Result: Sendable {
        let observation: [Float]
        let reward: Double
        let terminated: Bool
        let score: Int
        let maxTile: Int
    }

    private enum Direction: Int, Sendable {
        case up
        case right
        case down
        case left
    }
}
