import Foundation

struct MetroRunnerCatalog: Decodable, Sendable {
    let schemaVersion: Int
    let chunks: [String: Chunk]
    let sections: [String: Section]
    let entries: [Entry]
    let intro: String
    let fallback: String
    let repeatDistance: Double

    static let bundled: Result<MetroRunnerCatalog, Error> = Result {
        guard let url = Bundle.module.url(forResource: "metro-runner", withExtension: "json") else {
            throw GymnazoError.invalidConfiguration("MetroRunner route catalog is missing.")
        }
        let catalog = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        guard catalog.schemaVersion == 1,
            catalog.sections[catalog.intro] != nil, catalog.sections[catalog.fallback] != nil,
            catalog.entries.allSatisfy({ catalog.sections[$0.section] != nil })
        else {
            throw GymnazoError.invalidConfiguration("MetroRunner route catalog is invalid.")
        }
        return catalog
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
