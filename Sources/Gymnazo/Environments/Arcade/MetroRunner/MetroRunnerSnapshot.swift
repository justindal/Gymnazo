import Foundation

/// Geometry and pickup types represented by Metro Run.
public enum MetroRunnerEntityKind: String, Sendable, CaseIterable, Codable {
    case train, movingTrain = "moving_train", jump, roll, gate, ramp, coin
    case magnet, sneakers, multiplier, jetpack, mystery, key, platform
    case pillar, signal, dumpster, bush, powerbox, pogo, unknown
    case pickupAnchor = "pickup_anchor"

    var isHazard: Bool {
        [.train, .movingTrain, .jump, .roll, .gate, .platform, .pillar, .signal, .dumpster, .bush, .powerbox].contains(self)
    }
    var isSupport: Bool { [.train, .movingTrain, .platform].contains(self) }
    var isPower: Bool { [.magnet, .sneakers, .multiplier, .jetpack].contains(self) }
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
    public let diving: Bool
    public let pogo: Bool
    public let coins: Int
    public let score: Int
    public let mysteryBoxes: Int
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
    var source = ""
    var contacted = false
    var spawnGroup = 0
    var disableOnFlyover = false
    var preserveFlyoverDistance = 600.0
    var curveSpawner = ""
    var curveT = 0.0
    var spawnOrigin: SIMD3<Double>?
    var inAir = false

    var collisionBottom: Double { y + (kind == .roll ? 5 : 0) }
    var collisionTop: Double { y + (kind == .roll ? 24 : height) }

    func surface(at position: Double) -> Double {
        kind == .ramp ? y + height * max(0, min(1, (position - z) / (rampRun > 0 ? rampRun : depth))) : y + height
    }

    var snapshot: MetroRunnerEntitySnapshot {
        MetroRunnerEntitySnapshot(kind: kind, x: x, y: y, z: z, width: width, depth: depth,
                                  height: height, velocity: velocity, rampRun: rampRun)
    }
}
