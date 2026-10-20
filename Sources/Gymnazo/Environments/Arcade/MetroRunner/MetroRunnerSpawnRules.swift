import Foundation

struct MetroRunnerSpawnRules: Sendable {
    let rules: MetroRunnerRules
    let profile: MetroRunnerSpawnProfile
    var generator: SeededGenerator
    var lastGroupZ: Double?
    var lastPickupZ: [MetroRunnerEntityKind: Double] = [:]

    func requirementsMet(_ requirements: MetroRunnerRules.Requirements?) -> Bool {
        guard let requirements else { return true }
        let values = requirements.data.map { item in
            switch item.type {
            case "FeatureUnlocked": item.value.map { profile.unlockedFeatures.contains($0) } ?? false
            case "IsLootBoxFriendly": profile.lootBoxFriendly
            default: false
            }
        }
        return requirements.operation == "Or" ? values.contains(true) : values.allSatisfy { $0 }
    }

    func validEntries(at z: Double, group: Int) -> [MetroRunnerRules.SpawnEntry] {
        rules.spawnEntries.filter {
            $0.weight > 0 && $0.groups.contains(group) && requirementsMet($0.requirements)
                && z >= (lastPickupZ[$0.kind] ?? -.infinity) + $0.minimumDistance
        }
    }

    mutating func choose(at z: Double, group: Int, flying: Bool) -> MetroRunnerEntityKind? {
        guard !flying, z >= (lastGroupZ ?? -.infinity) + rules.spawnSpacing else { return nil }
        guard let kind = choose(from: validEntries(at: z, group: group)) else { return nil }
        lastGroupZ = z
        lastPickupZ[kind] = z
        return kind
    }

    mutating func choosePogo() -> MetroRunnerEntityKind? {
        choose(from: rules.pogoEntries.filter { $0.weight > 0 && requirementsMet($0.requirements) })
    }

    private mutating func choose(from entries: [MetroRunnerRules.SpawnEntry]) -> MetroRunnerEntityKind? {
        let total = entries.reduce(0) { $0 + $1.weight }
        guard total > 0 else { return nil }
        var ticket = Int.random(in: 0..<total, using: &generator)
        for entry in entries {
            ticket -= entry.weight
            if ticket < 0 { return entry.kind }
        }
        return nil
    }
}
