#if canImport(CoreGraphics)
    import CoreGraphics
#endif

public enum RenderMode: String, Sendable, Codable, CaseIterable {
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
    case collection([RenderOutput])
}

public struct EnvMetadata: Sendable, Codable, Equatable {
    public let renderModes: [RenderMode]
    public let renderFPS: Int?

    public init(renderModes: [RenderMode] = [], renderFPS: Int? = nil) {
        self.renderModes = renderModes
        self.renderFPS = renderFPS
    }

    static func bundled(id: String) -> EnvMetadata {
        switch id {
        case "FrozenLake", "FrozenLake8x8", "Taxi", "CliffWalking":
            return EnvMetadata(renderModes: [.human, .ansi, .rgbArray], renderFPS: 4)
        case "Blackjack":
            return EnvMetadata(renderModes: [.human, .rgbArray], renderFPS: 4)
        case "CartPole":
            return EnvMetadata(renderModes: [.human, .rgbArray], renderFPS: 50)
        case "MountainCar", "MountainCarContinuous", "Pendulum":
            return EnvMetadata(renderModes: [.human, .rgbArray], renderFPS: 30)
        case "Acrobot":
            return EnvMetadata(renderModes: [.human, .rgbArray], renderFPS: 15)
        case "LunarLander", "LunarLanderContinuous":
            return EnvMetadata(renderModes: [.human, .rgbArray], renderFPS: 50)
        case "CarRacing", "CarRacingDiscrete":
            return EnvMetadata(
                renderModes: [.human, .rgbArray, .statePixels],
                renderFPS: 50
            )
        default:
            return EnvMetadata()
        }
    }
}

extension Env {
    @discardableResult
    public mutating func render(mode: RenderMode) throws -> RenderOutput? {
        renderMode = mode
        return try render()
    }
}
