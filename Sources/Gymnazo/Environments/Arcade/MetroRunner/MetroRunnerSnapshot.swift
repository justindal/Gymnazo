import Foundation

/// Geometry and pickup types in observation-version-two one-hot order.
public enum MetroRunnerEntityKind: String, Sendable, CaseIterable {
    case train, movingTrain = "moving_train", jump, roll, gate, ramp, coin
    case magnet, sneakers, multiplier, jetpack, mystery, key, platform

    var isHazard: Bool { [.train, .movingTrain, .jump, .roll, .gate, .platform].contains(self) }
    var isSupport: Bool { [.train, .movingTrain, .platform].contains(self) }
    var powerDuration: Double? {
        switch self {
        case .magnet, .sneakers, .multiplier: 15
        case .jetpack: 12
        default: nil
        }
    }
}

/// Immutable geometry suitable for rendering outside an environment actor.
public struct MetroRunnerEntitySnapshot: Sendable, Equatable {
    public let kind: MetroRunnerEntityKind
    public let x: Double
    public let y: Double
    public let z: Double
    public let width: Double
    public let depth: Double
    public let height: Double
    public let velocity: Double
    public let rampRun: Double
}

/// A render-safe state snapshot in simulator world units.
public struct MetroRunnerSnapshot: Sendable, Equatable {
    public let x: Double
    public let y: Double
    public let surface: Double
    public let distance: Double
    public let speed: Double
    public let grounded: Bool
    public let rolling: Bool
    public let coins: Int
    public let score: Int
    public let boards: Int
    public let boardRemaining: Double
    public let powers: [MetroRunnerEntityKind: Double]
    public let dead: Bool
    public let entities: [MetroRunnerEntitySnapshot]
}

struct MetroRunnerEntity: Sendable, Equatable {
    var kind: MetroRunnerEntityKind
    var x: Double
    var y: Double
    var z: Double
    var width = 14.0
    var depth = 9.0
    var height = 12.0
    var velocity = 0.0
    var alive = true
    var rampRun = 0.0
    var carrierZ: Double?
    var motionFactor = 0.0

    func surface(at position: Double) -> Double {
        kind == .ramp ? y + height * max(0, min(1, (position - z) / (rampRun > 0 ? rampRun : depth))) : y + height
    }

    var snapshot: MetroRunnerEntitySnapshot {
        MetroRunnerEntitySnapshot(kind: kind, x: x, y: y, z: z, width: width, depth: depth,
                                  height: height, velocity: velocity, rampRun: rampRun)
    }
}
