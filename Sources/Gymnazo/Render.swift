#if canImport(CoreGraphics)
import CoreGraphics
import CoreFoundation
#endif

public enum RenderMode: String, Sendable {
    case human = "human"
    case rgbArray = "rgb_array"
    case ansi = "ansi"
}

public enum RenderOutput {
    case ansi(String)
#if canImport(CoreGraphics)
    case rgbArray(CGImage)
#endif
    case other(Any)
}

public extension Env {
    var renderMode: RenderMode? {
        get { renderMode.flatMap(RenderMode.init(rawValue:)) }
        set { renderMode = newValue?.rawValue }
    }

    @discardableResult
    mutating func render(mode: RenderMode) -> RenderOutput? {
        renderMode = mode
        return renderTyped()
    }

    @discardableResult
    func renderTyped() -> RenderOutput? {
        guard let value = render() else { return nil }
        if let s = value as? String {
            return .ansi(s)
        }
#if canImport(CoreGraphics)
        if let cf = value as CFTypeRef?, CFGetTypeID(cf) == CGImage.typeID {
            return .rgbArray(unsafeDowncast(cf, to: CGImage.self))
        }
#endif
        return .other(value)
    }
}
