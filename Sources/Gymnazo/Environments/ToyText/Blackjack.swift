//
// Blackjack.swift
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

/// Observation for the Blackjack environment.
///
/// Contains the player's current hand sum, the dealer's showing card value,
/// and whether the player has a usable ace.
public struct BlackjackObservation: Hashable, Sendable, Equatable {
    /// The player's current hand sum (4-21 typically, but can go higher when busting).
    public let playerSum: Int
    
    /// The dealer's face-up card value (1-10, where 1 is Ace).
    public let dealerCard: Int
    
    /// Whether the player has a usable ace (0 or 1).
    public let usableAce: Int
    
    public init(playerSum: Int, dealerCard: Int, usableAce: Int) {
        self.playerSum = playerSum
        self.dealerCard = dealerCard
        self.usableAce = usableAce
    }
}

/// The observation space for Blackjack, consisting of three discrete components.
public struct BlackjackObservationSpace: Space {
    public typealias T = BlackjackObservation
    
    public let playerSumSpace: Discrete
    public let dealerCardSpace: Discrete
    public let usableAceSpace: Discrete
    
    public init() {
        self.playerSumSpace = Discrete(n: 32)
        self.dealerCardSpace = Discrete(n: 11)
        self.usableAceSpace = Discrete(n: 2)
    }
    
    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> BlackjackObservation {
        let keys = MLX.split(key: key, into: 3)
        let playerSum = playerSumSpace.sample(key: keys[0])
        let dealerCard = dealerCardSpace.sample(key: keys[1])
        let usableAce = usableAceSpace.sample(key: keys[2])
        return BlackjackObservation(playerSum: playerSum, dealerCard: dealerCard, usableAce: usableAce)
    }
    
    public func contains(_ x: BlackjackObservation) -> Bool {
        playerSumSpace.contains(x.playerSum) &&
        dealerCardSpace.contains(x.dealerCard) &&
        usableAceSpace.contains(x.usableAce)
    }
}

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
/// The observation is a ``BlackjackObservation`` containing:
/// - `playerSum`: The player's current sum (4-21 typically).
/// - `dealerCard`: The value of the dealer's showing card (1-10 where 1 is ace).
/// - `usableAce`: Whether the player holds a usable ace (0 or 1).
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
    
    public typealias Observation = BlackjackObservation
    public typealias Action = Int
    public typealias ObservationSpace = BlackjackObservationSpace
    public typealias ActionSpace = Discrete
    
    public let action_space: Discrete
    public let observation_space: BlackjackObservationSpace
    public var spec: EnvSpec?
    public var render_mode: String?
    
    private let natural: Bool
    private let sab: Bool
    
    private var player: [Int] = []
    private var dealer: [Int] = []
    
    private var dealerTopCardSuit: String = "S"
    private var dealerTopCardValueStr: String = "A"
    
    private var playerCardSuits: [String] = []
    private var playerCardValueStrs: [String] = []
    
    private var _key: MLXArray?
    private var _renderKey: MLXArray?
    
#if canImport(SwiftUI)
    private var lastRGBFrame: CGImage?
#endif
    
    public init(
        render_mode: String? = nil,
        natural: Bool = false,
        sab: Bool = false
    ) {
        self.render_mode = render_mode
        self.natural = natural
        self.sab = sab
        
        self.action_space = Discrete(n: 2)
        self.observation_space = BlackjackObservationSpace()
    }
    
    private static func drawCard(key: MLXArray) -> Int {
        let index = MLX.randInt(low: 0, high: deck.count, key: key)
        return deck[Int(index.item(Int32.self))]
    }
    
    private static func drawHand(key: MLXArray) -> (hand: [Int], nextKey: MLXArray) {
        let (k1, temp) = MLX.split(key: key)
        let (k2, nextKey) = MLX.split(key: temp)
        let card1 = drawCard(key: k1)
        let card2 = drawCard(key: k2)
        return ([card1, card2], nextKey)
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
    
    private func generateCardDisplay(value: Int, key: MLXArray) -> (valueStr: String, suit: String, nextKey: MLXArray) {
        let (suitKey, temp) = MLX.split(key: key)
        let suitIndex = MLX.randInt(low: 0, high: 4, key: suitKey).item(Int32.self)
        let suit = Self.suits[Int(suitIndex)]
        
        let valueStr: String
        if value == 1 {
            valueStr = "A"
        } else if value == 10 {
            let (faceKey, nextKey) = MLX.split(key: temp)
            let faceIndex = MLX.randInt(low: 0, high: 3, key: faceKey).item(Int32.self)
            return (Self.faceCards[Int(faceIndex)], suit, nextKey)
        } else {
            valueStr = String(value)
        }
        
        return (valueStr, suit, temp)
    }
    
    private func getObs() -> BlackjackObservation {
        let playerSum = Self.sumHand(player)
        let usableAce = Self.usableAce(player) ? 1 : 0
        return BlackjackObservation(
            playerSum: playerSum,
            dealerCard: dealer[0],
            usableAce: usableAce
        )
    }
    
    public func reset(
        seed: UInt64? = nil,
        options: [String: Any]? = nil
    ) -> Reset<Observation> {
        if let seed {
            _key = MLX.key(seed)
            _renderKey = MLX.key(seed ^ 0x9E3779B97F4A7C15)
        } else {
            if _key == nil {
                let s = UInt64.random(in: 0...UInt64.max)
                _key = MLX.key(s)
            }
            if _renderKey == nil {
                let s = UInt64.random(in: 0...UInt64.max)
                _renderKey = MLX.key(s)
            }
        }
        
        guard var key = _key else {
            fatalError("Failed to initialize RNG key")
        }
        guard var renderKey = _renderKey else {
            fatalError("Failed to initialize render RNG key")
        }
        
        let (dealerHand, k1) = Self.drawHand(key: key)
        key = k1
        var (playerHand, k2) = Self.drawHand(key: key)
        key = k2
        
        while Self.sumHand(playerHand) < 12 {
            let (drawKey, nextKey) = MLX.split(key: key)
            key = nextKey
            playerHand.append(Self.drawCard(key: drawKey))
        }
        _key = key
        
        dealer = dealerHand
        player = playerHand
        
        let obs = getObs()
        
        let (dealerValStr, dealerSuit, displayKey) = generateCardDisplay(value: dealer[0], key: renderKey)
        dealerTopCardValueStr = dealerValStr
        dealerTopCardSuit = dealerSuit
        renderKey = displayKey
        
        playerCardSuits = []
        playerCardValueStrs = []
        for cardValue in player {
            let (valStr, suit, nextKey) = generateCardDisplay(value: cardValue, key: renderKey)
            playerCardValueStrs.append(valStr)
            playerCardSuits.append(suit)
            renderKey = nextKey
        }
        _renderKey = renderKey
        
        return Reset(obs: obs, info: [:])
    }
    
    public func step(_ action: Action) -> Step<Observation> {
        guard _key != nil else {
            fatalError("Call reset() before step()")
        }
        
        var terminated = false
        var reward: Double = 0.0
        
        if action == 1 {
            let (drawKey, nextKey) = MLX.split(key: _key!)
            _key = nextKey
            let newCard = Self.drawCard(key: drawKey)
            player.append(newCard)
            
            if _renderKey == nil {
                _renderKey = MLX.key(UInt64.random(in: 0...UInt64.max))
            }
            let (valStr, suit, k) = generateCardDisplay(value: newCard, key: _renderKey!)
            _renderKey = k
            playerCardValueStrs.append(valStr)
            playerCardSuits.append(suit)
            
            if Self.isBust(player) {
                terminated = true
                reward = -1.0
            }
        } else {
            terminated = true
            
            while Self.sumHand(dealer) < 17 {
                let (drawKey, nextKey) = MLX.split(key: _key!)
                _key = nextKey
                dealer.append(Self.drawCard(key: drawKey))
            }
            
            reward = Self.compare(Self.score(player), Self.score(dealer))
            
            if sab && Self.isNatural(player) && !Self.isNatural(dealer) {
                reward = 1.0
            } else if !sab && natural && Self.isNatural(player) && reward == 1.0 {
                reward = 1.5
            }
        }
        
        return Step(
            obs: getObs(),
            reward: reward,
            terminated: terminated,
            truncated: false,
            info: [:]
        )
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
        case "human", "rgb_array":
#if canImport(SwiftUI)
            let snapshot = currentSnapshot
            let result = DispatchQueue.main.sync {
                Blackjack.renderGUI(snapshot: snapshot, mode: mode)
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
    
#if canImport(SwiftUI)
    @MainActor
    private static func renderGUI(snapshot: BlackjackRenderSnapshot, mode: String) -> Any? {
        let view = BlackjackCanvasView(snapshot: snapshot)
        
        switch mode {
        case "human":
#if canImport(PlaygroundSupport)
            PlaygroundPage.current.setLiveView(view)
#else
            print("[Gymnazo] SwiftUI Canvas available via BlackjackCanvasView; integrate it into your app UI.")
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
        let bgColor = Color(red: 7/255, green: 99/255, blue: 36/255)
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
        let totalCardsWidth = CGFloat(cardCount) * cardWidth + CGFloat(max(0, cardCount - 1)) * spacing
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

