#if canImport(CoreGraphics)
    import CoreGraphics

    enum ArcadeImageRenderer {
        static func image(
            width: Int = 600,
            height: Int = 400,
            draw: (CGContext, CGSize) -> Void
        ) -> CGImage? {
            guard
                let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
                let context = CGContext(
                    data: nil,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: width * 4,
                    space: colorSpace,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                )
            else {
                return nil
            }

            draw(context, CGSize(width: width, height: height))
            return context.makeImage()
        }
    }
#endif
