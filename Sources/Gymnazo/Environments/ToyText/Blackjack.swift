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

/// Blackjack is a card game where the goal is to beat the dealer by obtaining cards
/// that sum to closer to 21 (without going over 21) than the dealer's cards.
///
/// ## Description
///
/// The game starts with the dealer having one face up and one face down card,
/// while the player has two face up cards. All cards are drawn from an infinite deck
/// (i.e., with replacement).
///
/// The card values are:
/// - Face cards (Jack, Queen, King) have a point value of 10.
/// - Aces can either count as 11 (called a 'usable ace') or 1.
/// - Numerical cards (2-10) have a value equal to their number.
///
/// The player can request additional cards (hit) until they decide to stop (stick)
/// or exceed 21 (bust, immediate loss).
///
/// After the player sticks, the dealer reveals their facedown card and draws cards
/// until their sum is 17 or greater. If the dealer goes bust, the player wins.
///
/// If neither the player nor the dealer busts, the outcome (win, lose, draw) is
/// decided by whose sum is closer to 21.
///
/// ## Action Space
///
/// The action is an `Int` in the range `{0, 1}`:
///
/// | Action | Meaning |
/// |--------|---------|
/// | 0      | Stick   |
/// | 1      | Hit     |
///
/// ## Observation Space
///
/// The observation is a `Tuple(Discrete(32), Discrete(11), Discrete(2))`:
/// - Player's current sum (0-31).
/// - Dealer's showing card (0-10, where 1 is ace).
/// - Whether the player holds a usable ace (0 or 1).
///
/// ## Starting State
///
/// Both player and dealer are dealt two cards from an infinite deck.
///
/// ## Rewards
///
/// - Win game: **+1**
/// - Lose game: **-1**
/// - Draw game: **0**
/// - Win with natural blackjack: **+1.5** (if `natural` is `true`) or **+1**
///
/// ## Episode End
///
/// The episode ends if:
///
/// **Termination:**
/// 1. The player hits and the sum of hand exceeds 21.
/// 2. The player sticks.
///
/// ## Arguments
///
/// - `render_mode`: The render mode (`"human"` or `"rgb_array"`).
/// - `natural`: Whether to give an additional reward (+1.5) for starting with a natural blackjack.
/// - `sab`: Whether to follow the exact rules from Sutton and Barto. If `true`, `natural` is ignored.
///
/// ## Version History
///
/// - v1: initial implementation
public final class Blackjack: Env {
    private static let deck: [Int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    public static var metadata: [String: Any] {
        [
            "render_modes": ["human", "rgb_array"],
            "render_fps": 4,
        ]
    }

    public let actionSpace: any Space
    public let observationSpace: any Space
    public var spec: EnvSpec?
    public var renderMode: RenderMode?

    private let natural: Bool
    private let sab: Bool

    private var player: [Int] = []
    private var dealer: [Int] = []

    private var dealerTopCardSuit: String = "S"
    private var dealerTopCardValueStr: String = "A"

    private var playerCardSuits: [String] = []
    private var playerCardValueStrs: [String] = []

    private var generator: SeededGenerator?
    private var renderGenerator: SeededGenerator?

    #if canImport(SwiftUI)
        private var lastRGBFrame: CGImage?
    #endif

    private func toInt(_ action: MLXArray) -> Int {
        Int(action.singletonValue(Int32.self))
    }

    public init(
        renderMode: RenderMode? = nil,
        natural: Bool = false,
        sab: Bool = false
    ) {
        self.renderMode = renderMode
        self.natural = natural
        self.sab = sab

        self.actionSpace = Discrete(n: 2)
        self.observationSpace = Tuple(Discrete(n: 32), Discrete(n: 11), Discrete(n: 2))
    }

    private static func drawCard(using generator: inout SeededGenerator) -> Int {
        deck[Int.random(in: deck.indices, using: &generator)]
    }

    private static func drawHand(using generator: inout SeededGenerator) -> [Int] {
        [drawCard(using: &generator), drawCard(using: &generator)]
    }

    private static func usableAce(_ hand: [Int]) -> Bool {
        hand.contains(1) && hand.reduce(0, +) + 10 <= 21
    }

    private static func sumHand(_ hand: [Int]) -> Int {
        if usableAce(hand) {
            return hand.reduce(0, +) + 10
        }
        return hand.reduce(0, +)
    }

    private static func isBust(_ hand: [Int]) -> Bool {
        sumHand(hand) > 21
    }

    private static func score(_ hand: [Int]) -> Int {
        isBust(hand) ? 0 : sumHand(hand)
    }

    private static func isNatural(_ hand: [Int]) -> Bool {
        hand.sorted() == [1, 10]
    }

    private static func compare(_ a: Int, _ b: Int) -> Double {
        if a > b { return 1.0 }
        if a < b { return -1.0 }
        return 0.0
    }

    private static let suits = ["C", "D", "H", "S"]
    private static let faceCards = ["J", "Q", "K"]

    private func generateCardDisplay(
        value: Int,
        using generator: inout SeededGenerator
    ) -> (valueStr: String, suit: String) {
        let suit = Self.suits[Int.random(in: Self.suits.indices, using: &generator)]

        let valueStr: String
        if value == 1 {
            valueStr = "A"
        } else if value == 10 {
            let face = Self.faceCards[Int.random(in: Self.faceCards.indices, using: &generator)]
            return (face, suit)
        } else {
            valueStr = String(value)
        }

        return (valueStr, suit)
    }

    private func getObs() -> MLXArray {
        let playerSum = Self.sumHand(player)
        let usableAce = Self.usableAce(player) ? 1 : 0
        return MLXArray([Int32(playerSum), Int32(dealer[0]), Int32(usableAce)])
    }

    public func reset(
        seed: UInt64? = nil,
        options: EnvOptions? = nil
    ) throws -> Reset {
        if let seed {
            generator = SeededGenerator(seed: seed)
            renderGenerator = SeededGenerator(seed: seed ^ 0x9E37_79B9_7F4A_7C15)
        } else {
            if generator == nil {
                generator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
            }
            if renderGenerator == nil {
                renderGenerator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
            }
        }

        guard var generator else {
            throw GymnazoError.invalidState("Failed to initialize random generator")
        }
        guard var renderGenerator else {
            throw GymnazoError.invalidState("Failed to initialize render random generator")
        }

        let dealerHand = Self.drawHand(using: &generator)
        var playerHand = Self.drawHand(using: &generator)

        while Self.sumHand(playerHand) < 12 {
            playerHand.append(Self.drawCard(using: &generator))
        }
        self.generator = generator

        dealer = dealerHand
        player = playerHand

        let obs = getObs()

        let (dealerValStr, dealerSuit) = generateCardDisplay(
            value: dealer[0], using: &renderGenerator)
        dealerTopCardValueStr = dealerValStr
        dealerTopCardSuit = dealerSuit

        playerCardSuits = []
        playerCardValueStrs = []
        for cardValue in player {
            let (valStr, suit) = generateCardDisplay(
                value: cardValue,
                using: &renderGenerator
            )
            playerCardValueStrs.append(valStr)
            playerCardSuits.append(suit)
        }
        self.renderGenerator = renderGenerator

        return Reset(obs: obs, info: [:])
    }

    public func step(_ action: MLXArray) throws -> Step {
        let a = toInt(action)
        guard actionSpace.contains(action) else {
            throw GymnazoError.invalidAction("Invalid action: \(a)")
        }
        guard var generator else {
            throw GymnazoError.stepBeforeReset
        }

        var terminated = false
        var reward: Double = 0.0

        if a == 1 {
            let newCard = Self.drawCard(using: &generator)
            player.append(newCard)

            if renderGenerator == nil {
                renderGenerator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
            }
            guard var renderGenerator else {
                throw GymnazoError.invalidState("Missing render random generator")
            }
            let (valStr, suit) = generateCardDisplay(
                value: newCard,
                using: &renderGenerator
            )
            self.renderGenerator = renderGenerator
            playerCardValueStrs.append(valStr)
            playerCardSuits.append(suit)

            if Self.isBust(player) {
                terminated = true
                reward = -1.0
            }
        } else {
            terminated = true

            while Self.sumHand(dealer) < 17 {
                dealer.append(Self.drawCard(using: &generator))
            }

            reward = Self.compare(Self.score(player), Self.score(dealer))

            if sab && Self.isNatural(player) && !Self.isNatural(dealer) {
                reward = 1.0
            } else if !sab && natural && Self.isNatural(player) && reward == 1.0 {
                reward = 1.5
            }
        }

        self.generator = generator
        return Step(
            obs: getObs(),
            reward: reward,
            terminated: terminated,
            truncated: false,
            info: [:]
        )
    }

    @discardableResult
    public func render() throws -> RenderOutput? {
        guard let mode = renderMode else {
            if let specId = spec?.id {
                print(
                    "[Gymnazo] render() called without renderMode. Set renderMode when creating \(specId)."
                )
            }
            return nil
        }

        switch mode {
        case .human:
            #if canImport(SwiftUI)
                return .other(currentSnapshot)
            #else
                return nil
            #endif
        case .rgbArray:
            return nil
        case .ansi, .statePixels:
            print("[Gymnazo] Unsupported renderMode \(mode.rawValue).")
            return nil
        }
    }

    #if canImport(SwiftUI)
        @MainActor
        private static func renderGUI(snapshot: BlackjackRenderSnapshot, mode: RenderMode)
            -> CGImage?
        {
            let view = BlackjackCanvasView(snapshot: snapshot)

            switch mode {
            case .human:
                #if canImport(PlaygroundSupport)
                    PlaygroundPage.current.setLiveView(view)
                #else
                    print(
                        "[Gymnazo] SwiftUI Canvas available via BlackjackCanvasView; integrate it into your app UI."
                    )
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

        public var currentSnapshot: BlackjackRenderSnapshot {
            BlackjackRenderSnapshot(
                playerSum: Self.sumHand(player),
                dealerCard: dealer.first ?? 0,
                usableAce: Self.usableAce(player),
                dealerCardSuit: dealerTopCardSuit,
                dealerCardValueStr: dealerTopCardValueStr,
                playerCardValueStrs: playerCardValueStrs,
                playerCardSuits: playerCardSuits
            )
        }

        @MainActor
        public func humanView() -> BlackjackCanvasView {
            BlackjackCanvasView(snapshot: currentSnapshot)
        }
    #endif
}

#if canImport(SwiftUI)
    /// Snapshot of Blackjack game state used by SwiftUI renderers.
    public struct BlackjackRenderSnapshot: Sendable, Equatable {
        public let playerSum: Int
        public let dealerCard: Int
        public let usableAce: Bool
        public let dealerCardSuit: String
        public let dealerCardValueStr: String
        public let playerCardValueStrs: [String]
        public let playerCardSuits: [String]
    }

    /// SwiftUI Canvas view that renders a snapshot of Blackjack.
    public struct BlackjackCanvasView: View {
        public let snapshot: BlackjackRenderSnapshot

        public init(snapshot: BlackjackRenderSnapshot) {
            self.snapshot = snapshot
        }

        private let cardWidth: CGFloat = 70
        private let cardHeight: CGFloat = 100
        private let cornerRadius: CGFloat = 8

        public var body: some View {
            Canvas { context, size in
                drawBackground(context: &context, size: size)
                drawDealerSection(context: &context, size: size)
                drawPlayerSection(context: &context, size: size)
            }
            .frame(width: 600, height: 500)
        }

        private func drawBackground(context: inout GraphicsContext, size: CGSize) {
            let bgColor = Color(red: 7 / 255, green: 99 / 255, blue: 36 / 255)
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(bgColor))
        }

        private func drawDealerSection(context: inout GraphicsContext, size: CGSize) {
            let spacing: CGFloat = 20
            let dealerText = "Dealer: \(snapshot.dealerCard)"

            context.draw(
                Text(dealerText)
                    .font(.system(size: 24, weight: .bold))
                    .foregroundColor(.white),
                at: CGPoint(x: spacing + 60, y: spacing + 12)
            )

            let cardY = spacing + 50
            let card1X = size.width / 2 - cardWidth - spacing / 2
            let card2X = size.width / 2 + spacing / 2

            drawCard(
                context: &context,
                x: card1X,
                y: cardY,
                value: snapshot.dealerCardValueStr,
                suit: snapshot.dealerCardSuit,
                faceUp: true
            )

            drawCard(
                context: &context,
                x: card2X,
                y: cardY,
                value: "?",
                suit: "",
                faceUp: false
            )
        }

        private func drawPlayerSection(context: inout GraphicsContext, size: CGSize) {
            let spacing: CGFloat = 20
            let playerY = cardHeight + 100

            let playerLabel = "Player: \(snapshot.playerSum)"
            context.draw(
                Text(playerLabel)
                    .font(.system(size: 24, weight: .bold))
                    .foregroundColor(.white),
                at: CGPoint(x: spacing + 70, y: playerY)
            )

            if snapshot.usableAce {
                context.draw(
                    Text("(usable ace)")
                        .font(.system(size: 16))
                        .foregroundColor(.white.opacity(0.8)),
                    at: CGPoint(x: spacing + 180, y: playerY)
                )
            }

            let cardCount = snapshot.playerCardValueStrs.count
            let totalCardsWidth =
                CGFloat(cardCount) * cardWidth + CGFloat(max(0, cardCount - 1)) * spacing
            let startX = (size.width - totalCardsWidth) / 2
            let cardY = playerY + 30

            for i in 0..<cardCount {
                let x = startX + CGFloat(i) * (cardWidth + spacing)
                drawCard(
                    context: &context,
                    x: x,
                    y: cardY,
                    value: snapshot.playerCardValueStrs[i],
                    suit: snapshot.playerCardSuits[i],
                    faceUp: true
                )
            }
        }

        private func drawCard(
            context: inout GraphicsContext,
            x: CGFloat,
            y: CGFloat,
            value: String,
            suit: String,
            faceUp: Bool
        ) {
            let rect = CGRect(x: x, y: y, width: cardWidth, height: cardHeight)
            let path = Path(roundedRect: rect, cornerRadius: cornerRadius)

            if faceUp {
                context.fill(path, with: .color(.white))
                context.stroke(path, with: .color(.black), lineWidth: 1)

                let suitColor: Color = (suit == "H" || suit == "D") ? .red : .black
                let suitSymbol: String
                switch suit {
                case "H": suitSymbol = "♥"
                case "D": suitSymbol = "♦"
                case "C": suitSymbol = "♣"
                case "S": suitSymbol = "♠"
                default: suitSymbol = ""
                }

                context.draw(
                    Text(value)
                        .font(.system(size: 20, weight: .bold))
                        .foregroundColor(suitColor),
                    at: CGPoint(x: x + 15, y: y + 15)
                )

                context.draw(
                    Text(suitSymbol)
                        .font(.system(size: 14))
                        .foregroundColor(suitColor),
                    at: CGPoint(x: x + 15, y: y + 35)
                )

                context.draw(
                    Text(suitSymbol)
                        .font(.system(size: 32))
                        .foregroundColor(suitColor),
                    at: CGPoint(x: x + cardWidth / 2, y: y + cardHeight / 2)
                )
            } else {
                let backColor = Color(red: 0.1, green: 0.2, blue: 0.6)
                context.fill(path, with: .color(backColor))
                context.stroke(path, with: .color(.white), lineWidth: 2)

                let patternRect = rect.insetBy(dx: 5, dy: 5)
                let patternPath = Path(roundedRect: patternRect, cornerRadius: cornerRadius - 2)
                context.stroke(patternPath, with: .color(.white.opacity(0.5)), lineWidth: 1)
            }
        }
    }
#endif
