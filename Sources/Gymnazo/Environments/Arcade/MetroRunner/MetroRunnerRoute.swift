import Foundation

struct MetroRunnerRoute: Sendable {
    let catalog: MetroRunnerCatalog
    let fixedSection: String?
    var generator: SeededGenerator
    var pending: [String] = []
    var section = ""
    var lastPlacement: [String: Double] = [:]
    var unique: Set<Int> = []
    var alternating: [String: Int] = [:]

    init(catalog: MetroRunnerCatalog, seed: UInt64, section: String?) throws {
        self.catalog = catalog
        self.fixedSection = section
        self.generator = SeededGenerator(seed: seed)
        try queue(section ?? catalog.intro)
    }

    mutating func next(at distance: Double) throws -> String {
        if pending.isEmpty {
            if let fixedSection {
                try queue(fixedSection)
            } else {
                let candidates = eligible(at: distance)
                if !candidates.isEmpty {
                    let choice = generator.weightedIndex(candidates.map { max(0, catalog.entries[$0].weight) }) ?? 0
                    let index = candidates[choice]
                    let entry = catalog.entries[index]
                    if entry.unique { unique.insert(index) }
                    try queue(entry.section)
                } else {
                    try queue(catalog.fallback)
                }
            }
        }
        guard !pending.isEmpty else {
            throw GymnazoError.invalidConfiguration("MetroRunner section produced no chunks.")
        }
        return pending.removeFirst()
    }

    func eligible(at distance: Double) -> [Int] {
        let z = distance.rounded(.towardZero)
        let candidates = catalog.entries.indices.filter { index in
            let entry = catalog.entries[index]
            return (entry.minimum <= 0 || entry.minimum < z)
                && (entry.maximum <= 0 || z < entry.maximum)
                && (!entry.unique || !unique.contains(index))
                && (lastPlacement[entry.section].map { $0 + catalog.repeatDistance <= z } ?? true)
        }
        let priority = max(0, candidates.map { catalog.entries[$0].priority }.max() ?? 0)
        return candidates.filter { catalog.entries[$0].priority == priority }
    }

    mutating func placed(at end: Double) {
        lastPlacement[section] = end
    }

    private mutating func queue(_ name: String) throws {
        pending.append(contentsOf: try expand(name))
        section = name
    }

    mutating func expand(_ name: String) throws -> [String] {
        guard let value = catalog.sections[name] else {
            throw GymnazoError.invalidConfiguration("Unknown MetroRunner section: \(name).")
        }
        if value.kind == "SectionCompositeConfig" {
            var result: [String] = []
            for child in value.sections { result.append(contentsOf: try expand(child)) }
            return result
        }
        guard !value.chunks.isEmpty else { return [] }
        switch value.kind {
        case "SectionSequentialConfig": return value.chunks
        case "SectionRandomOneConfig":
            return [value.chunks[Int.random(in: value.chunks.indices, using: &generator)]]
        case "SectionAlternatingOneConfig":
            let index = alternating[name] ?? Int.random(in: value.chunks.indices, using: &generator)
            alternating[name] = (index + 1) % value.chunks.count
            return [value.chunks[index]]
        default:
            throw GymnazoError.invalidConfiguration("Unsupported MetroRunner section iterator: \(value.kind).")
        }
    }
}
