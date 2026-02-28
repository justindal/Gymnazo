import MLXNN

public enum ActivationConfig: String, Sendable, Codable {
    case relu
    case tanh

    public func make() -> any UnaryLayer {
        switch self {
        case .relu:
            return ReLU()
        case .tanh:
            return Tanh()
        }
    }
}
