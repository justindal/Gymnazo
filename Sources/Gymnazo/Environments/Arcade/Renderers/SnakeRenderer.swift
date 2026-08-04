#if canImport(SwiftUI)
    import SwiftUI

    /// A scalable SwiftUI renderer for ``SnakeSnapshot``.
    public struct SnakeView: View {
        public let snapshot: SnakeSnapshot

        /// Creates a renderer for an immutable environment snapshot.
        public init(snapshot: SnakeSnapshot) {
            self.snapshot = snapshot
        }

        public var body: some View {
            Canvas { context, size in
                let cellWidth = size.width / CGFloat(snapshot.columns)
                let cellHeight = size.height / CGFloat(snapshot.rows)

                context.fill(
                    Path(CGRect(origin: .zero, size: size)),
                    with: .color(Color(red: 0.06, green: 0.08, blue: 0.07))
                )

                for (index, cell) in snapshot.body.enumerated() {
                    let rect = cellRect(
                        cell,
                        cellWidth: cellWidth,
                        cellHeight: cellHeight
                    ).insetBy(dx: 1.5, dy: 1.5)
                    context.fill(
                        Path(roundedRect: rect, cornerRadius: 4),
                        with: .color(index == 0 ? .mint : .green)
                    )
                }

                if let food = snapshot.food {
                    let rect = cellRect(
                        food,
                        cellWidth: cellWidth,
                        cellHeight: cellHeight
                    ).insetBy(dx: cellWidth * 0.18, dy: cellHeight * 0.18)
                    context.fill(Path(ellipseIn: rect), with: .color(.red))
                }

                context.draw(
                    Text("Score: \(snapshot.score)")
                        .font(.headline)
                        .foregroundStyle(.white),
                    at: CGPoint(x: 12, y: 12),
                    anchor: .topLeading
                )
            }
            .aspectRatio(1, contentMode: .fit)
            .accessibilityLabel("Snake environment, score \(snapshot.score)")
        }

        private func cellRect(
            _ cell: SnakeCell,
            cellWidth: CGFloat,
            cellHeight: CGFloat
        ) -> CGRect {
            CGRect(
                x: CGFloat(cell.x) * cellWidth,
                y: CGFloat(snapshot.rows - 1 - cell.y) * cellHeight,
                width: cellWidth,
                height: cellHeight
            )
        }
    }
#endif

#if canImport(CoreGraphics)
    import CoreGraphics

    enum SnakeImageRenderer {
        static func image(snapshot: SnakeSnapshot) -> CGImage? {
            ArcadeImageRenderer.image { context, size in
                context.setFillColor(
                    CGColor(red: 0.06, green: 0.08, blue: 0.07, alpha: 1)
                )
                context.fill(CGRect(origin: .zero, size: size))

                let cellWidth = size.width / CGFloat(snapshot.columns)
                let cellHeight = size.height / CGFloat(snapshot.rows)
                for (index, cell) in snapshot.body.enumerated() {
                    let rect = CGRect(
                        x: CGFloat(cell.x) * cellWidth + 1.5,
                        y: CGFloat(cell.y) * cellHeight + 1.5,
                        width: cellWidth - 3,
                        height: cellHeight - 3
                    )
                    context.setFillColor(
                        index == 0
                            ? CGColor(red: 0.25, green: 0.95, blue: 0.72, alpha: 1)
                            : CGColor(red: 0.16, green: 0.72, blue: 0.3, alpha: 1)
                    )
                    context.fill(rect)
                }

                if let food = snapshot.food {
                    let insetX = cellWidth * 0.18
                    let insetY = cellHeight * 0.18
                    let rect = CGRect(
                        x: CGFloat(food.x) * cellWidth + insetX,
                        y: CGFloat(food.y) * cellHeight + insetY,
                        width: cellWidth - insetX * 2,
                        height: cellHeight - insetY * 2
                    )
                    context.setFillColor(
                        CGColor(red: 0.92, green: 0.18, blue: 0.2, alpha: 1)
                    )
                    context.fillEllipse(in: rect)
                }
            }
        }
    }
#endif
