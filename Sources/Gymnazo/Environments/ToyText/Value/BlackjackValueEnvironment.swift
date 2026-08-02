public struct BlackjackValueEnvironment: ValueEnvironment {
    private static let deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    public let valueObservationSpace: TupleValueSpace
    public let valueActionSpace: DiscreteValueSpace

    private let natural: Bool
    private let sab: Bool
    private var player: [Int] = []
    private var dealer: [Int] = []
    private var generator: SeededGenerator?

    public init(natural: Bool = false, sab: Bool = false) throws {
        valueObservationSpace = TupleValueSpace([
            AnyValueSpace(try DiscreteValueSpace(n: 32)),
            AnyValueSpace(try DiscreteValueSpace(n: 11)),
            AnyValueSpace(try DiscreteValueSpace(n: 2)),
        ])
        valueActionSpace = try DiscreteValueSpace(n: 2)
        self.natural = natural
        self.sab = sab
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<[SpaceValue]> {
        guard options == nil || options?.isEmpty == true else {
            throw GymnazoError.invalidConfiguration("Blackjack does not support reset options.")
        }
        prepareGenerator(seed: seed)
        guard var generator else {
            throw GymnazoError.invalidState("Random generator is unavailable.")
        }
        dealer = [drawCard(using: &generator), drawCard(using: &generator)]
        player = [drawCard(using: &generator), drawCard(using: &generator)]
        while Self.sum(player) < 12 {
            player.append(drawCard(using: &generator))
        }
        self.generator = generator
        return ValueReset(observation: observation)
    }

    public mutating func stepValue(_ action: Int) throws -> ValueStep<[SpaceValue]> {
        guard valueActionSpace.containsValue(action) else {
            throw GymnazoError.invalidAction("Invalid action: \(action)")
        }
        guard var generator else {
            throw GymnazoError.stepBeforeReset
        }

        var reward = 0.0
        var terminated = false
        if action == 1 {
            player.append(drawCard(using: &generator))
            if Self.isBust(player) {
                reward = -1
                terminated = true
            }
        } else {
            terminated = true
            while Self.sum(dealer) < 17 {
                dealer.append(drawCard(using: &generator))
            }
            reward = Self.compare(Self.score(player), Self.score(dealer))
            if sab, Self.isNatural(player), !Self.isNatural(dealer) {
                reward = 1
            } else if !sab, natural, Self.isNatural(player), reward == 1 {
                reward = 1.5
            }
        }
        self.generator = generator
        return ValueStep(
            observation: observation,
            reward: reward,
            terminated: terminated,
            truncated: false
        )
    }

    public mutating func close() {}

    private var observation: [SpaceValue] {
        [
            .discrete(Self.sum(player)),
            .discrete(dealer[0]),
            .discrete(Self.usableAce(player) ? 1 : 0),
        ]
    }

    private mutating func prepareGenerator(seed: UInt64?) {
        if let seed {
            generator = SeededGenerator(seed: seed)
        } else if generator == nil {
            generator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
        }
    }

    private func drawCard(using generator: inout SeededGenerator) -> Int {
        Self.deck[Int.random(in: Self.deck.indices, using: &generator)]
    }

    private static func usableAce(_ hand: [Int]) -> Bool {
        hand.contains(1) && hand.reduce(0, +) + 10 <= 21
    }

    private static func sum(_ hand: [Int]) -> Int {
        hand.reduce(0, +) + (usableAce(hand) ? 10 : 0)
    }

    private static func isBust(_ hand: [Int]) -> Bool {
        sum(hand) > 21
    }

    private static func score(_ hand: [Int]) -> Int {
        isBust(hand) ? 0 : sum(hand)
    }

    private static func isNatural(_ hand: [Int]) -> Bool {
        hand.sorted() == [1, 10]
    }

    private static func compare(_ lhs: Int, _ rhs: Int) -> Double {
        lhs == rhs ? 0 : (lhs > rhs ? 1 : -1)
    }
}
