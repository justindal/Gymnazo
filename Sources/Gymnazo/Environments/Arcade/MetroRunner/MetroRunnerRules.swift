import Foundation

struct MetroRunnerRules: Decodable, Sendable {
    let durations: [String: [Double]]
    let sneakersCurve: MetroRunnerCurve
    let spawnSpacing: Double
    let spawnEntries: [SpawnEntry]
    let pogoEntries: [SpawnEntry]
    let jetpack: Jetpack
    let pogo: Pogo
    let curves: [String: CoinCurve]
    let coinSpawners: [String: CoinSpawner]

    func duration(_ kind: MetroRunnerEntityKind, level: Int) -> Double {
        durations[kind.rawValue]?[level] ?? 0
    }

    struct Jetpack: Decodable, Sendable {
        let height: Double
        let speedFactor: Double
        let adjustmentTime: Double
        let coinOffset: Double
        let coinSpacing: Double
        let coinHeight: Double
        let coinsPerNode: Int
        let laneDuration: Double
    }

    struct Pogo: Decodable, Sendable {
        let height: Double
        let distance: Double
        let curve: MetroRunnerCurve
        let rows: Int
        let coinHeight: Double
        let coinStart: Double
        let coinEnd: Double
        let rewardHeight: Double
        let rewardOffset: Double
    }

    struct CoinCurve: Decodable, Sendable {
        let curve: MetroRunnerCurve
        let zScale: Double
        let yScale: Double
        let fixedOffset: Double
    }

    struct CoinSpawner: Decodable, Sendable {
        let curve: String
        let baseCount: Int
        let scalesWithSpeed: Bool
        let pivot: Double
        let inAir: Bool

        func count(speedModifier: Double) -> Int {
            scalesWithSpeed ? Int(floor((Double(baseCount) - 0.5) * speedModifier)) + 1 : baseCount
        }
    }

    struct SpawnEntry: Decodable, Sendable {
        let kind: MetroRunnerEntityKind
        let weight: Int
        let groups: [Int]
        let minimumDistance: Double
        let requirements: Requirements?
    }

    struct Requirements: Decodable, Sendable {
        let data: [Requirement]
        let operation: String

        enum CodingKeys: String, CodingKey {
            case data
            case operation = "operator"
        }
    }

    struct Requirement: Decodable, Sendable {
        let type: String
        let value: String?
    }

    func coinPosition(_ name: String, t: Double, speedModifier: Double) -> SIMD3<Double> {
        guard let spawner = coinSpawners[name], let curve = curves[spawner.curve] else { return .zero }
        let width = curve.zScale + curve.fixedOffset
        let pivot = (spawner.pivot - 0.5) * width
        let z = spawner.scalesWithSpeed
            ? pivot + (t - 0.5) * curve.fixedOffset + speedModifier * ((t - 0.5) * curve.zScale - pivot)
            : (t - 0.5) * width
        return SIMD3(0, curve.curve.value(at: t) * curve.yScale, z)
    }
}
