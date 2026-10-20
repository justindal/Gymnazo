import Foundation

struct MetroRunnerCatalog: Decodable, Sendable {
    let schemaVersion: Int
    let chunks: [String: Chunk]
    let sections: [String: Section]
    let entries: [Entry]
    let intro: String
    let fallback: String
    let repeatDistance: Double
    let rules: MetroRunnerRules

    static let bundled: Result<MetroRunnerCatalog, Error> = Result {
        guard let url = Bundle.module.url(forResource: "metro-runner", withExtension: "json") else {
            throw GymnazoError.invalidConfiguration("MetroRunner route catalog is missing.")
        }
        let catalog = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        try catalog.validate()
        return catalog
    }

    func validate() throws {
        guard schemaVersion == 2,
            sections[intro] != nil, sections[fallback] != nil,
            entries.allSatisfy({ sections[$0.section] != nil }),
            ["magnet", "sneakers", "multiplier", "jetpack"].allSatisfy({
                rules.durations[$0]?.count == 7
            }),
            rules.durations.values.flatMap({ $0 }).allSatisfy({ $0.isFinite && $0 > 0 }),
            rules.jetpack.adjustmentTime > 0,
            rules.jetpack.coinSpacing > 0, rules.pogo.rows > 0, rules.pogo.distance > 0,
            rules.coinSpawners.values.allSatisfy({ rules.curves[$0.curve] != nil && $0.baseCount > 0 })
        else {
            throw GymnazoError.invalidConfiguration("MetroRunner route catalog is invalid.")
        }
        let supported: Set<String> = [
            "obstacle_barrier_jump", "obstacle_barrier_roll", "obstacle_barrier_standard",
            "obstacle_pillar", "obstacle_lightSignal", "obstacle_dumpster", "obstacle_bush", "obstacle_powerBox",
        ]
        func check(_ node: Node) throws {
            if node.asset.hasPrefix("obstacle_"), !supported.contains(node.asset) {
                throw GymnazoError.invalidConfiguration("Unsupported MetroRunner obstacle asset: \(node.asset).")
            }
            if node.pickupType == 1, let name = node.spawner,
                name != "CoinSingleSpawner", rules.coinSpawners[name] == nil {
                throw GymnazoError.invalidConfiguration("Unsupported MetroRunner coin spawner: \(name).")
            }
            for child in node.children { try check(child) }
        }
        for chunk in chunks.values { try check(chunk.root) }
    }

    struct Chunk: Decodable, Sendable {
        let length: Double
        let root: Node
    }

    struct Node: Decodable, Sendable {
        let name: String
        let active: Bool
        let position: [Double]
        let rotation: [Double]
        let scale: [Double]
        let asset: String
        let placer: Bool
        let lanes: [Double]?
        let mirror: Double?
        let randomChild: Double?
        let movingSpeed: Double
        let spawn: Bool
        let coin: Bool
        let coinCount: Int
        let spawner: String?
        let pickupType: Int?
        let spawnGroup: Int
        let disableOnFlyover: Bool
        let preserveFlyoverDistance: Double
        let colliders: [Collider]
        let children: [Node]
    }

    struct Collider: Decodable, Sendable {
        let center: [Double]
        let size: [Double]
    }

    struct Section: Decodable, Sendable {
        let kind: String
        let chunks: [String]
        let sections: [String]
    }

    struct Entry: Decodable, Sendable {
        let section: String
        let weight: Double
        let minimum: Double
        let maximum: Double
        let priority: Int
        let unique: Bool
    }
}
