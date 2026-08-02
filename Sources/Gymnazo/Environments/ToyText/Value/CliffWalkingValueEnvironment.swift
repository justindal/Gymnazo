/// Typed Swift-state CliffWalking environment.
public struct CliffWalkingValueEnvironment: ValueEnvironment {
    private static let rows = 4
    private static let columns = 12
    private static let start = 36
    private static let goal = 47
    private static let deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    public let valueObservationSpace: DiscreteValueSpace
    public let valueActionSpace: DiscreteValueSpace

    private let slippery: Bool
    private var state = Self.start
    private var generator: SeededGenerator?

    /// Creates a grid with optional slippery transitions.
    ///
    /// - Parameter isSlippery: Whether actions may drift to neighboring directions.
    /// - Throws: If the value-space configuration fails.
    public init(isSlippery: Bool = false) throws {
        valueObservationSpace = try DiscreteValueSpace(n: Self.rows * Self.columns)
        valueActionSpace = try DiscreteValueSpace(n: Self.deltas.count)
        slippery = isSlippery
    }

    /// Resets the agent to the fixed start tile.
    ///
    /// - Parameters:
    ///   - seed: Optional deterministic reset seed.
    ///   - options: CliffWalking does not support reset options.
    /// - Returns: The start state and reset metadata.
    /// - Throws: If options are supplied.
    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Int> {
        guard options == nil || options?.isEmpty == true else {
            throw GymnazoError.invalidConfiguration("CliffWalking does not support reset options.")
        }
        prepareGenerator(seed: seed)
        state = Self.start
        return ValueReset(observation: state, info: ["prob": .double(1)])
    }

    /// Moves the agent and applies cliff or goal rewards.
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

        let actions =
            slippery
            ? [(action + 3) % 4, action, (action + 1) % 4]
            : [action]
        let index = Int.random(in: actions.indices, using: &generator)
        self.generator = generator
        let selected = actions[index]
        let probability = 1 / Double(actions.count)
        let row = state / Self.columns
        let column = state % Self.columns
        let delta = Self.deltas[selected]
        let nextRow = min(max(row + delta.0, 0), Self.rows - 1)
        let nextColumn = min(max(column + delta.1, 0), Self.columns - 1)
        let next = nextRow * Self.columns + nextColumn

        if nextRow == Self.rows - 1, (1..<(Self.columns - 1)).contains(nextColumn) {
            state = Self.start
            return ValueStep(
                observation: state,
                reward: -100,
                terminated: false,
                truncated: false,
                info: ["prob": .double(probability)]
            )
        }

        state = next
        return ValueStep(
            observation: state,
            reward: -1,
            terminated: state == Self.goal,
            truncated: false,
            info: ["prob": .double(probability)]
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
}
