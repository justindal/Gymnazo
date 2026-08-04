#if canImport(SwiftUI)
    import SwiftUI

    /// A scalable SwiftUI renderer for ``FlappyBirdSnapshot``.
    public struct FlappyBirdView: View {
        public let snapshot: FlappyBirdSnapshot

        /// Creates a renderer for an immutable environment snapshot.
        public init(snapshot: FlappyBirdSnapshot) {
            self.snapshot = snapshot
        }

        public var body: some View {
            Canvas { context, size in
                let scaleX = size.width
                let scaleY = size.height

                context.fill(
                    Path(CGRect(origin: .zero, size: size)),
                    with: .color(Color(red: 0.48, green: 0.78, blue: 0.9))
                )

                for pipe in snapshot.pipes {
                    let x = CGFloat(pipe.x - FlappyBirdModel.pipeWidth / 2) * scaleX
                    let width = CGFloat(FlappyBirdModel.pipeWidth) * scaleX
                    let gapBottom = CGFloat(pipe.gapY - FlappyBirdModel.gapHeight / 2)
                    let gapTop = CGFloat(pipe.gapY + FlappyBirdModel.gapHeight / 2)
                    let upperHeight = (1 - gapTop) * scaleY
                    let lowerY = (1 - gapBottom) * scaleY

                    context.fill(
                        Path(CGRect(x: x, y: 0, width: width, height: upperHeight)),
                        with: .color(.green)
                    )
                    context.fill(
                        Path(CGRect(x: x, y: lowerY, width: width, height: size.height - lowerY)),
                        with: .color(.green)
                    )
                }

                let birdSize = CGSize(
                    width: CGFloat(FlappyBirdModel.birdWidth) * scaleX,
                    height: CGFloat(FlappyBirdModel.birdHeight) * scaleY
                )
                let birdOrigin = CGPoint(
                    x: CGFloat(FlappyBirdModel.birdX) * scaleX - birdSize.width / 2,
                    y: (1 - CGFloat(snapshot.birdY)) * scaleY - birdSize.height / 2
                )
                context.fill(
                    Path(ellipseIn: CGRect(origin: birdOrigin, size: birdSize)),
                    with: .color(.yellow)
                )

                context.draw(
                    Text("Score: \(snapshot.score)")
                        .font(.headline)
                        .foregroundStyle(.primary),
                    at: CGPoint(x: 16, y: 16),
                    anchor: .topLeading
                )
            }
            .aspectRatio(3 / 2, contentMode: .fit)
            .accessibilityLabel("Flappy Bird environment, score \(snapshot.score)")
        }
    }
#endif

#if canImport(CoreGraphics)
    import CoreGraphics

    enum FlappyBirdImageRenderer {
        static func image(snapshot: FlappyBirdSnapshot) -> CGImage? {
            ArcadeImageRenderer.image { context, size in
                context.setFillColor(
                    CGColor(red: 0.48, green: 0.78, blue: 0.9, alpha: 1)
                )
                context.fill(CGRect(origin: .zero, size: size))

                context.setFillColor(
                    CGColor(red: 0.18, green: 0.65, blue: 0.22, alpha: 1)
                )
                for pipe in snapshot.pipes {
                    let x = CGFloat(pipe.x - FlappyBirdModel.pipeWidth / 2) * size.width
                    let width = CGFloat(FlappyBirdModel.pipeWidth) * size.width
                    let gapBottom = CGFloat(pipe.gapY - FlappyBirdModel.gapHeight / 2)
                    let gapTop = CGFloat(pipe.gapY + FlappyBirdModel.gapHeight / 2)
                    context.fill(
                        CGRect(x: x, y: 0, width: width, height: gapBottom * size.height)
                    )
                    context.fill(
                        CGRect(
                            x: x,
                            y: gapTop * size.height,
                            width: width,
                            height: (1 - gapTop) * size.height
                        )
                    )
                }

                let birdSize = CGSize(
                    width: CGFloat(FlappyBirdModel.birdWidth) * size.width,
                    height: CGFloat(FlappyBirdModel.birdHeight) * size.height
                )
                let birdOrigin = CGPoint(
                    x: CGFloat(FlappyBirdModel.birdX) * size.width - birdSize.width / 2,
                    y: CGFloat(snapshot.birdY) * size.height - birdSize.height / 2
                )
                context.setFillColor(
                    CGColor(red: 1, green: 0.82, blue: 0.08, alpha: 1)
                )
                context.fillEllipse(in: CGRect(origin: birdOrigin, size: birdSize))
            }
        }
    }
#endif
