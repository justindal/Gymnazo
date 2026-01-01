#if canImport(CoreGraphics)
import CoreGraphics
import MLX

public extension CGImage {
    /// Convert a CGImage to an MLXArray with shape [height, width, 3] and dtype uint8.
    func toMLXArray() -> MLXArray? {
        let width = self.width
        let height = self.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let totalBytes = height * bytesPerRow
        
        var pixelData = [UInt8](repeating: 0, count: totalBytes)
        
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            return nil
        }
        
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }
        
        context.draw(self, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        var rgbData = [UInt8](repeating: 0, count: height * width * 3)
        for y in 0..<height {
            for x in 0..<width {
                let srcIdx = (y * width + x) * 4
                let dstIdx = (y * width + x) * 3
                rgbData[dstIdx] = pixelData[srcIdx]
                rgbData[dstIdx + 1] = pixelData[srcIdx + 1]
                rgbData[dstIdx + 2] = pixelData[srcIdx + 2]
            }
        }
        
        return MLXArray(rgbData).reshaped([height, width, 3]).asType(.uint8)
    }
    
    /// Create a scaled version of this image.
    func scaled(to size: CGSize) -> CGImage? {
        let width = Int(size.width)
        let height = Int(size.height)
        
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            return nil
        }
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }
        
        context.interpolationQuality = .high
        context.draw(self, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return context.makeImage()
    }
}
#endif

