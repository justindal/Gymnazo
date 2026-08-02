/// Typed Swift-state Taxi environment.
public struct TaxiValueEnvironment: ValueEnvironment {
    private static let map = [
        "+---------+",
        "|R: | : :G|",
        "| : | : : |",
        "| : : : : |",
        "| | : | : |",
        "|Y| : |B: |",
        "+---------+",
    ].map { Array($0.utf8) }
    private static let locations = [(0, 0), (0, 4), (4, 0), (4, 3)]

    public let valueObservationSpace: DiscreteValueSpace
    public let valueActionSpace: DiscreteValueSpace

    private let rainy: Bool
    private let ficklePassenger: Bool
    private let initialStates: [Int]
    private var state = 0
    private var fickleStep = false
    private var generator: SeededGenerator?

    /// Creates a Taxi environment with optional rainy and fickle dynamics.
    ///
    /// - Parameters:
    ///   - isRainy: Whether movement may drift to neighboring directions.
    ///   - ficklePassenger: Whether the passenger destination may change.
    /// - Throws: If the value-space configuration fails.
    public init(isRainy: Bool = false, ficklePassenger: Bool = false) throws {
        valueObservationSpace = try DiscreteValueSpace(n: 500)
        valueActionSpace = try DiscreteValueSpace(n: 6)
        rainy = isRainy
        self.ficklePassenger = ficklePassenger
        initialStates = (0..<5).flatMap { row in
            (0..<5).flatMap { column in
                (0..<4).flatMap { passenger in
                    (0..<4).compactMap { destination in
                        passenger == destination
                            ? nil
                            : Self.encode(
                                row: row,
                                column: column,
                                passenger: passenger,
                                destination: destination
                            )
                    }
                }
            }
        }
    }

    /// Resets the taxi and samples a passenger task.
    ///
    /// - Parameters:
    ///   - seed: Optional deterministic reset seed.
    ///   - options: Taxi does not support reset options.
    /// - Returns: The initial state and action-mask metadata.
    /// - Throws: If options are supplied or sampling fails.
    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Int> {
        guard options == nil || options?.isEmpty == true else {
            throw GymnazoError.invalidConfiguration("Taxi does not support reset options.")
        }
        prepareGenerator(seed: seed)
        guard var generator else {
            throw GymnazoError.invalidState("Random generator is unavailable.")
        }
        state = initialStates[Int.random(in: initialStates.indices, using: &generator)]
        fickleStep = ficklePassenger && generator.uniform(in: 0...1) < 0.3
        self.generator = generator
        return ValueReset(
            observation: state,
            info: [
                "prob": .double(1),
                "action_mask": actionMaskValue(state),
            ]
        )
    }

    /// Applies a movement, pickup, or drop-off action.
    ///
    /// - Parameter action: Movement, pickup, or drop-off action index.
    /// - Returns: The next state and step result.
    /// - Throws: If the action is invalid or the environment is not reset.
    public mutating func stepValue(_ action: Int) throws -> ValueStep<Int> {
        guard valueActionSpace.containsValue(action) else {
            throw GymnazoError.invalidAction("Invalid action: \(action)")
        }
        guard var generator else {
            throw GymnazoError.stepBeforeReset
        }

        let previous = Self.decode(state)
        var next = previous
        var reward = -1.0
        var terminated = false
        var probability = 1.0

        if action < 4 {
            let moves = Self.moves(for: action)
            let selected: (Int, Int)
            if rainy,
                Self.canMove(
                    row: previous.row,
                    column: previous.column,
                    movement: moves.intended
                )
            {
                let choices = [moves.intended, moves.left, moves.right]
                let weights = [0.8, 0.1, 0.1]
                let index = generator.weightedIndex(weights) ?? 0
                selected = choices[index]
                probability = weights[index]
            } else {
                selected = moves.intended
            }
            let position = Self.move(
                row: previous.row,
                column: previous.column,
                movement: selected
            )
            next.row = position.0
            next.column = position.1
        } else if action == 4 {
            if previous.passenger < 4,
                (previous.row, previous.column) == Self.locations[previous.passenger]
            {
                next.passenger = 4
            } else {
                reward = -10
            }
        } else if previous.passenger == 4,
            (previous.row, previous.column) == Self.locations[previous.destination]
        {
            next.passenger = previous.destination
            reward = 20
            terminated = true
        } else if previous.passenger == 4,
            let location = Self.locations.firstIndex(where: {
                $0 == (previous.row, previous.column)
            })
        {
            next.passenger = location
        } else {
            reward = -10
        }

        if ficklePassenger,
            fickleStep,
            previous.passenger == 4,
            next.row != previous.row || next.column != previous.column
        {
            fickleStep = false
            let destinations = Self.locations.indices.filter { $0 != previous.destination }
            next.destination =
                destinations[
                    Int.random(in: destinations.indices, using: &generator)
                ]
        }

        state = Self.encode(
            row: next.row,
            column: next.column,
            passenger: next.passenger,
            destination: next.destination
        )
        self.generator = generator
        return ValueStep(
            observation: state,
            reward: reward,
            terminated: terminated,
            truncated: false,
            info: [
                "prob": .double(probability),
                "action_mask": actionMaskValue(state),
            ]
        )
    }

    public mutating func close() {}

    /// Returns valid action flags for a taxi state.
    ///
    /// - Parameter state: Encoded taxi state.
    /// - Returns: Boolean flags for each action in action-space order.
    public func actionMask(for state: Int) -> [Bool] {
        let value = Self.decode(state)
        var mask = [Bool](repeating: false, count: 6)
        mask[0] = value.row < 4
        mask[1] = value.row > 0
        mask[2] = value.column < 4 && Self.canMoveEast(row: value.row, column: value.column)
        mask[3] = value.column > 0 && Self.canMoveWest(row: value.row, column: value.column)
        mask[4] =
            value.passenger < 4
            && (value.row, value.column) == Self.locations[value.passenger]
        mask[5] =
            value.passenger == 4
            && Self.locations.contains(where: { $0 == (value.row, value.column) })
        return mask
    }

    private func actionMaskValue(_ state: Int) -> SpaceValue {
        .multiBinary(actionMask(for: state))
    }

    private mutating func prepareGenerator(seed: UInt64?) {
        if let seed {
            generator = SeededGenerator(seed: seed)
        } else if generator == nil {
            generator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
        }
    }

    private static func encode(
        row: Int,
        column: Int,
        passenger: Int,
        destination: Int
    ) -> Int {
        (((row * 5) + column) * 5 + passenger) * 4 + destination
    }

    private static func decode(
        _ state: Int
    ) -> (row: Int, column: Int, passenger: Int, destination: Int) {
        var value = state
        let destination = value % 4
        value /= 4
        let passenger = value % 5
        value /= 5
        let column = value % 5
        return (value / 5, column, passenger, destination)
    }

    private static func moves(
        for action: Int
    ) -> (intended: (Int, Int), left: (Int, Int), right: (Int, Int)) {
        switch action {
        case 0:
            return ((1, 0), (0, -1), (0, 1))
        case 1:
            return ((-1, 0), (0, -1), (0, 1))
        case 2:
            return ((0, 1), (1, 0), (-1, 0))
        default:
            return ((0, -1), (1, 0), (-1, 0))
        }
    }

    private static func move(
        row: Int,
        column: Int,
        movement: (Int, Int)
    ) -> (Int, Int) {
        guard canMove(row: row, column: column, movement: movement) else {
            return (row, column)
        }
        return (
            min(max(row + movement.0, 0), 4),
            min(max(column + movement.1, 0), 4)
        )
    }

    private static func canMove(
        row: Int,
        column: Int,
        movement: (Int, Int)
    ) -> Bool {
        if movement.1 > 0 {
            return column < 4 && canMoveEast(row: row, column: column)
        }
        if movement.1 < 0 {
            return column > 0 && canMoveWest(row: row, column: column)
        }
        return (0...4).contains(row + movement.0)
    }

    private static func canMoveEast(row: Int, column: Int) -> Bool {
        map[1 + row][2 * column + 2] == UInt8(ascii: ":")
    }

    private static func canMoveWest(row: Int, column: Int) -> Bool {
        map[1 + row][2 * column] == UInt8(ascii: ":")
    }
}
