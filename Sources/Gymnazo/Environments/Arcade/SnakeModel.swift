struct SnakeModel: Sendable {
    static let columns = 12
    static let rows = 12
    static let observationCount = columns * rows + 4

    private(set) var body: [SnakeCell] = []
    private(set) var food: SnakeCell?
    private(set) var score = 0

    private var direction = Direction.right
    private var generator: SeededGenerator?
    private var needsReset = true

    mutating func reset(seed: UInt64?) -> [Float] {
        if let seed {
            generator = SeededGenerator(seed: seed)
        } else if generator == nil {
            generator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
        }

        let centerY = Self.rows / 2
        body = [
            SnakeCell(x: 5, y: centerY),
            SnakeCell(x: 4, y: centerY),
            SnakeCell(x: 3, y: centerY),
        ]
        direction = .right
        score = 0
        food = nextFood()
        needsReset = false
        return observation
    }

    mutating func step(action: Int) throws -> Result {
        guard !needsReset else {
            throw GymnazoError.stepBeforeReset
        }

        direction = direction.turned(action: action)
        let head = body[0]
        let nextHead = SnakeCell(
            x: head.x + direction.offset.x,
            y: head.y + direction.offset.y
        )
        let ateFood = nextHead == food
        let occupiedBody = ateFood ? body[...] : body.dropLast()
        let collided =
            nextHead.x < 0
            || nextHead.x >= Self.columns
            || nextHead.y < 0
            || nextHead.y >= Self.rows
            || occupiedBody.contains(nextHead)

        if collided {
            needsReset = true
            return Result(
                observation: observation,
                reward: -1,
                terminated: true,
                score: score,
                won: false
            )
        }

        body.insert(nextHead, at: 0)
        if ateFood {
            score += 1
            food = nextFood()
        } else {
            body.removeLast()
        }

        let won = food == nil
        needsReset = won
        return Result(
            observation: observation,
            reward: ateFood ? 1 : -0.01,
            terminated: won,
            score: score,
            won: won
        )
    }

    var snapshot: SnakeSnapshot {
        SnakeSnapshot(
            columns: Self.columns,
            rows: Self.rows,
            body: body,
            food: food,
            score: score
        )
    }

    private var observation: [Float] {
        var values = [Float](repeating: 0, count: Self.observationCount)
        for cell in body.dropFirst() {
            values[cell.y * Self.columns + cell.x] = 1 / 3
        }
        if let head = body.first {
            values[head.y * Self.columns + head.x] = 2 / 3
        }
        if let food {
            values[food.y * Self.columns + food.x] = 1
        }
        values[Self.columns * Self.rows + direction.rawValue] = 1
        return values
    }

    private mutating func nextFood() -> SnakeCell? {
        let occupied = Set(body)
        let available = (0..<Self.rows).flatMap { y in
            (0..<Self.columns).compactMap { x in
                let cell = SnakeCell(x: x, y: y)
                return occupied.contains(cell) ? nil : cell
            }
        }
        guard !available.isEmpty, var generator else { return nil }
        let index = Int.random(in: 0..<available.count, using: &generator)
        self.generator = generator
        return available[index]
    }

    struct Result: Sendable {
        let observation: [Float]
        let reward: Double
        let terminated: Bool
        let score: Int
        let won: Bool
    }

    private enum Direction: Int, Sendable {
        case up
        case right
        case down
        case left

        var offset: SnakeCell {
            switch self {
            case .up: SnakeCell(x: 0, y: 1)
            case .right: SnakeCell(x: 1, y: 0)
            case .down: SnakeCell(x: 0, y: -1)
            case .left: SnakeCell(x: -1, y: 0)
            }
        }

        func turned(action: Int) -> Direction {
            switch action {
            case 1:
                Direction(rawValue: (rawValue + 1) % 4) ?? self
            case 2:
                Direction(rawValue: (rawValue + 3) % 4) ?? self
            default:
                self
            }
        }
    }
}
