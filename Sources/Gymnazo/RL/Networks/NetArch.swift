//
//  NetArch.swift
//  Gymnazo
//

/// Architecture specification for actor-critic networks.
///
/// - Parameters:
///   - actor: Policy (actor) network hidden sizes.
///   - critic: Critic network hidden sizes.
public enum NetArch: Sendable, Equatable, Codable {
    case shared([Int])
    case separate(actor: [Int], critic: [Int])
}

extension NetArch {
    public var actor: [Int] {
        switch self {
        case .shared(let dims):
            return dims
        case .separate(let actor, _):
            return actor
        }
    }

    public var critic: [Int] {
        switch self {
        case .shared(let dims):
            return dims
        case .separate(_, let critic):
            return critic
        }
    }
}
