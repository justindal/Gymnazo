import Foundation

#if canImport(SwiftUI)
    import SwiftUI

    public struct TwentyFortyEightView: View {
        public let snapshot: TwentyFortyEightSnapshot

        public init(snapshot: TwentyFortyEightSnapshot) {
            self.snapshot = snapshot
        }

        public var body: some View {
            Canvas { context, size in
                let board = boardLayout(size: size)
                context.fill(
                    Path(CGRect(origin: .zero, size: size)),
                    with: .color(Color(red: 0.93, green: 0.88, blue: 0.78))
                )
                context.fill(
                    Path(roundedRect: board.frame, cornerRadius: board.cornerRadius),
                    with: .color(Color(red: 0.70, green: 0.63, blue: 0.54))
                )

                for row in 0..<boardSize {
                    for column in 0..<boardSize {
                        let tileIndex = row * boardSize + column
                        let value = tileIndex < snapshot.tiles.count ? snapshot.tiles[tileIndex] : 0
                        let rect = board.rect(column: column, row: row)
                        context.fill(
                            Path(roundedRect: rect, cornerRadius: board.tileCornerRadius),
                            with: .color(tileColor(for: value))
                        )

                        guard value > 0 else { continue }
                        let foreground =
                            value <= 4 ? Color(red: 0.35, green: 0.30, blue: 0.25) : .white
                        context.draw(
                            Text("\(value)")
                                .font(
                                    .system(
                                        size: board.fontSize(for: value), weight: .bold,
                                        design: .rounded)
                                )
                                .foregroundStyle(foreground),
                            at: CGPoint(x: rect.midX, y: rect.midY),
                            anchor: .center
                        )
                    }
                }

                context.draw(
                    Text("Score  \(snapshot.score)")
                        .font(.system(.headline, design: .rounded).bold())
                        .foregroundStyle(Color(red: 0.35, green: 0.30, blue: 0.25)),
                    at: CGPoint(x: board.frame.minX, y: board.frame.minY - 14),
                    anchor: .bottomLeading
                )

                if snapshot.isGameOver {
                    context.fill(
                        Path(roundedRect: board.frame, cornerRadius: board.cornerRadius),
                        with: .color(Color.white.opacity(0.72))
                    )
                    context.draw(
                        Text("Game Over")
                            .font(
                                .system(
                                    size: board.gameOverFontSize, weight: .bold, design: .rounded)
                            )
                            .foregroundStyle(Color(red: 0.35, green: 0.30, blue: 0.25)),
                        at: CGPoint(x: board.frame.midX, y: board.frame.midY),
                        anchor: .center
                    )
                }
            }
            .aspectRatio(1, contentMode: .fit)
            .accessibilityLabel(
                "2048 environment, score \(snapshot.score), highest tile \(snapshot.maxTile)"
            )
        }

        private let boardSize = TwentyFortyEightModel.boardSize

        private func boardLayout(size: CGSize) -> BoardLayout {
            let boardWidth = min(size.width * 0.9, size.height * 0.9)
            let origin = CGPoint(
                x: (size.width - boardWidth) / 2,
                y: (size.height - boardWidth) / 2 + 12
            )
            let spacing = boardWidth * 0.025
            let tileSize = (boardWidth - spacing * CGFloat(boardSize + 1)) / CGFloat(boardSize)
            return BoardLayout(
                frame: CGRect(x: origin.x, y: origin.y, width: boardWidth, height: boardWidth),
                tileSize: tileSize,
                spacing: spacing,
                cornerRadius: boardWidth * 0.035,
                tileCornerRadius: boardWidth * 0.018,
                gameOverFontSize: boardWidth * 0.1
            )
        }

        private func tileColor(for value: Int) -> Color {
            switch value {
            case 0: Color(red: 0.80, green: 0.75, blue: 0.67)
            case 2: Color(red: 0.93, green: 0.89, blue: 0.82)
            case 4: Color(red: 0.93, green: 0.87, blue: 0.73)
            case 8: Color(red: 0.95, green: 0.68, blue: 0.38)
            case 16: Color(red: 0.95, green: 0.57, blue: 0.32)
            case 32: Color(red: 0.95, green: 0.42, blue: 0.27)
            case 64: Color(red: 0.92, green: 0.26, blue: 0.16)
            case 128: Color(red: 0.91, green: 0.78, blue: 0.39)
            case 256: Color(red: 0.91, green: 0.76, blue: 0.29)
            case 512: Color(red: 0.90, green: 0.70, blue: 0.20)
            case 1024: Color(red: 0.86, green: 0.60, blue: 0.12)
            default: Color(red: 0.78, green: 0.48, blue: 0.08)
            }
        }

        private struct BoardLayout {
            let frame: CGRect
            let tileSize: CGFloat
            let spacing: CGFloat
            let cornerRadius: CGFloat
            let tileCornerRadius: CGFloat
            let gameOverFontSize: CGFloat

            func rect(column: Int, row: Int) -> CGRect {
                CGRect(
                    x: frame.minX + spacing + CGFloat(column) * (tileSize + spacing),
                    y: frame.minY + spacing + CGFloat(row) * (tileSize + spacing),
                    width: tileSize,
                    height: tileSize
                )
            }

            func fontSize(for value: Int) -> CGFloat {
                switch value {
                case 0..<100: tileSize * 0.38
                case 100..<1000: tileSize * 0.3
                case 1000..<10000: tileSize * 0.23
                default: tileSize * 0.18
                }
            }
        }
    }
#endif

#if canImport(CoreGraphics)
    import CoreGraphics

    enum TwentyFortyEightImageRenderer {
        static func image(snapshot: TwentyFortyEightSnapshot) -> CGImage? {
            ArcadeImageRenderer.image(width: 600, height: 600) { context, size in
                context.setFillColor(CGColor(red: 0.93, green: 0.88, blue: 0.78, alpha: 1))
                context.fill(CGRect(origin: .zero, size: size))

                let boardSize = TwentyFortyEightModel.boardSize
                let boardWidth = size.width * 0.9
                let origin = CGPoint(x: size.width * 0.05, y: size.height * 0.08)
                let spacing = boardWidth * 0.025
                let tileSize = (boardWidth - spacing * CGFloat(boardSize + 1)) / CGFloat(boardSize)
                let boardFrame = CGRect(
                    x: origin.x,
                    y: origin.y,
                    width: boardWidth,
                    height: boardWidth
                )

                context.setFillColor(CGColor(red: 0.70, green: 0.63, blue: 0.54, alpha: 1))
                context.fill(boardFrame)

                for row in 0..<boardSize {
                    for column in 0..<boardSize {
                        let tileIndex = row * boardSize + column
                        let value = tileIndex < snapshot.tiles.count ? snapshot.tiles[tileIndex] : 0
                        let rect = CGRect(
                            x: boardFrame.minX + spacing + CGFloat(column) * (tileSize + spacing),
                            y: boardFrame.minY + spacing + CGFloat(row) * (tileSize + spacing),
                            width: tileSize,
                            height: tileSize
                        )
                        context.setFillColor(tileColor(for: value))
                        context.fill(rect)
                    }
                }

                if snapshot.isGameOver {
                    context.setFillColor(CGColor(gray: 1, alpha: 0.72))
                    context.fill(boardFrame)
                }
            }
        }

        private static func tileColor(for value: Int) -> CGColor {
            switch value {
            case 0: CGColor(red: 0.80, green: 0.75, blue: 0.67, alpha: 1)
            case 2: CGColor(red: 0.93, green: 0.89, blue: 0.82, alpha: 1)
            case 4: CGColor(red: 0.93, green: 0.87, blue: 0.73, alpha: 1)
            case 8: CGColor(red: 0.95, green: 0.68, blue: 0.38, alpha: 1)
            case 16: CGColor(red: 0.95, green: 0.57, blue: 0.32, alpha: 1)
            case 32: CGColor(red: 0.95, green: 0.42, blue: 0.27, alpha: 1)
            case 64: CGColor(red: 0.92, green: 0.26, blue: 0.16, alpha: 1)
            case 128: CGColor(red: 0.91, green: 0.78, blue: 0.39, alpha: 1)
            case 256: CGColor(red: 0.91, green: 0.76, blue: 0.29, alpha: 1)
            case 512: CGColor(red: 0.90, green: 0.70, blue: 0.20, alpha: 1)
            case 1024: CGColor(red: 0.86, green: 0.60, blue: 0.12, alpha: 1)
            default: CGColor(red: 0.78, green: 0.48, blue: 0.08, alpha: 1)
            }
        }
    }
#endif
