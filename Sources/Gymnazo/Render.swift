#if canImport(CoreGraphics)
import CoreGraphics
#endif

public enum RenderMode: String, Sendable {
    case human = "human"
    case rgbArray = "rgb_array"
    case ansi = "ansi"
    case statePixels = "state_pixels"
}

public enum RenderOutput {
    case ansi(String)
#if canImport(CoreGraphics)
    case rgbArray(CGImage)
    case statePixels(CGImage)
#endif
    case other(any Sendable)
}

public extension Env {
    @discardableResult
    mutating func render(mode: RenderMode) throws -> RenderOutput? {
        renderMode = mode
        return try render()
    }
}
