/// Typed Swift-state FrozenLake environment.
public struct FrozenLakeValueEnvironment: ValueEnvironment {
    private enum Direction: Int, CaseIterable {
        case left
        case down
        case right
        case up
    }

    private struct Transition: Sendable {
        let probability: Double
        let nextState: Int
        let reward: Double
        let terminated: Bool
    }

    public let valueObservationSpace: DiscreteValueSpace
    public let valueActionSpace: DiscreteValueSpace

    private let transitions: [[[Transition]]]
    private let initialStates: [Int]
    private var state = 0
    private var generator: SeededGenerator?

    /// Creates a FrozenLake map with optional slipping and custom success rate.
    ///
    /// - Parameters:
    ///   - desc: Optional custom map rows.
    ///   - mapName: Bundled map name when desc is nil.
    ///   - isSlippery: Whether actions drift to neighboring directions.
    ///   - successRate: Probability of taking the selected direction.
    /// - Throws: If the map or success rate is invalid.
    public init(
        desc: [String]? = nil,
        mapName: String = "4x4",
        isSlippery: Bool = true,
        successRate: Float = 1 / 3
    ) throws {
        guard (0...1).contains(successRate) else {
            throw GymnazoError.invalidConfiguration(
                "successRate must be between zero and one."
            )
        }
        let map: [String]
        if let desc {
            map = desc
        } else if let named = FrozenLake.MAPS[mapName] {
            map = named
        } else {
            throw GymnazoError.invalidMap("Unknown map name '\(mapName)'")
        }
        let cells = try Self.validate(map)
        let rows = cells.count
        let columns = cells[0].count
        valueObservationSpace = try DiscreteValueSpace(n: rows * columns)
        valueActionSpace = try DiscreteValueSpace(n: Direction.allCases.count)
        initialStates = cells.indices.flatMap { row in
            cells[row].indices.compactMap { column in
                cells[row][column] == "S" ? row * columns + column : nil
            }
        }
        transitions = Self.makeTransitions(
            cells: cells,
            slippery: isSlippery,
            successRate: Double(successRate)
        )
    }

    /// Resets the agent to one of the map's start tiles.
    ///
    /// - Parameters:
    ///   - seed: Optional deterministic reset seed.
    ///   - options: FrozenLake does not support reset options.
    /// - Returns: The start state and reset metadata.
    /// - Throws: If options are supplied or the map has no start tiles.
    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Int> {
        guard options == nil || options?.isEmpty == true else {
            throw GymnazoError.invalidConfiguration("FrozenLake does not support reset options.")
        }
        prepareGenerator(seed: seed)
        guard var generator else {
            throw GymnazoError.invalidState("Random generator is unavailable.")
        }
        state = initialStates[Int.random(in: initialStates.indices, using: &generator)]
        self.generator = generator
        return ValueReset(observation: state, info: ["prob": .double(1)])
    }

    /// Samples a transition for the selected compass action.
    ///
    /// - Parameter action: Direction index from zero through three.
    /// - Returns: The next state and step result.
    /// - Throws: If the action is invalid or the environment is not reset.
    public mutating func stepValue(_ action: Int) throws -> ValueStep<Int> {
        guard valueActionSpace.containsValue(action) else {
            throw GymnazoError.invalidAction("Invalid action: \(action)")
        }
        guard var generator else {
            throw GymnazoError.stepBeforeReset
        }
        let candidates = transitions[state][action]
        guard let index = generator.weightedIndex(candidates.map(\.probability)) else {
            throw GymnazoError.invalidState("Transition probabilities are invalid.")
        }
        self.generator = generator
        let transition = candidates[index]
        state = transition.nextState
        return ValueStep(
            observation: state,
            reward: transition.reward,
            terminated: transition.terminated,
            truncated: false,
            info: ["prob": .double(transition.probability)]
        )
    }

    public mutating func close() {}

    private mutating func prepareGenerator(seed: UInt64?) {
        if let seed {
            generator = SeededGenerator(seed: seed)
        } else if generator == nil {
            generator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
        }
    }

    private static func validate(_ map: [String]) throws -> [[Character]] {
        guard let first = map.first, !first.isEmpty else {
            throw GymnazoError.invalidMap("Map description is empty")
        }
        let columns = first.count
        let allowed: Set<Character> = ["S", "F", "H", "G"]
        var startCount = 0
        var goalCount = 0
        var cells: [[Character]] = []
        cells.reserveCapacity(map.count)
        for (index, row) in map.enumerated() {
            guard row.count == columns else {
                throw GymnazoError.invalidMap(
                    "Row \(index) has \(row.count) columns, expected \(columns)"
                )
            }
            let values = Array(row)
            guard values.allSatisfy(allowed.contains) else {
                throw GymnazoError.invalidMap("Map contains an invalid tile")
            }
            startCount += values.filter { $0 == "S" }.count
            goalCount += values.filter { $0 == "G" }.count
            cells.append(values)
        }
        guard startCount > 0 else {
            throw GymnazoError.invalidMap("Map requires at least one start tile (S)")
        }
        guard goalCount > 0 else {
            throw GymnazoError.invalidMap("Map requires at least one goal tile (G)")
        }
        return cells
    }

    private static func makeTransitions(
        cells: [[Character]],
        slippery: Bool,
        successRate: Double
    ) -> [[[Transition]]] {
        let rows = cells.count
        let columns = cells[0].count
        let failRate = (1 - successRate) / 2
        var result = Array(
            repeating: Array(repeating: [Transition](), count: Direction.allCases.count),
            count: rows * columns
        )

        for row in cells.indices {
            for column in cells[row].indices {
                let state = row * columns + column
                for action in Direction.allCases {
                    if cells[row][column] == "G" || cells[row][column] == "H" {
                        result[state][action.rawValue] = [
                            Transition(
                                probability: 1,
                                nextState: state,
                                reward: 0,
                                terminated: true
                            )
                        ]
                        continue
                    }
                    let directions: [(Direction, Double)]
                    if slippery {
                        directions = [
                            (Direction(rawValue: (action.rawValue + 3) % 4)!, failRate),
                            (action, successRate),
                            (Direction(rawValue: (action.rawValue + 1) % 4)!, failRate),
                        ]
                    } else {
                        directions = [(action, 1)]
                    }
                    result[state][action.rawValue] = directions.map { direction, probability in
                        let next = move(
                            row: row,
                            column: column,
                            direction: direction,
                            rows: rows,
                            columns: columns
                        )
                        let tile = cells[next.row][next.column]
                        return Transition(
                            probability: probability,
                            nextState: next.row * columns + next.column,
                            reward: tile == "G" ? 1 : 0,
                            terminated: tile == "G" || tile == "H"
                        )
                    }
                }
            }
        }
        return result
    }

    private static func move(
        row: Int,
        column: Int,
        direction: Direction,
        rows: Int,
        columns: Int
    ) -> (row: Int, column: Int) {
        switch direction {
        case .left:
            return (row, max(column - 1, 0))
        case .down:
            return (min(row + 1, rows - 1), column)
        case .right:
            return (row, min(column + 1, columns - 1))
        case .up:
            return (max(row - 1, 0), column)
        }
    }
}
