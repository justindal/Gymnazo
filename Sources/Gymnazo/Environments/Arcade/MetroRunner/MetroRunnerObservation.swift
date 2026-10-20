import Foundation

/// Versioned structured state, independent of route identifiers and future random choices.
public enum MetroRunnerObservation {
    public static let version = 5
    public static let maxEntities = 64
    public static let entityFeatures = 31
    public static let playerFeatures = 41
    public static let flatCount = maxEntities * entityFeatures + maxEntities + playerFeatures
    static let kinds: [MetroRunnerEntityKind] = [
        .train, .movingTrain, .jump, .roll, .gate, .ramp, .coin, .magnet, .sneakers, .multiplier,
        .jetpack, .mystery, .key, .platform, .pillar, .signal, .dumpster, .bush, .powerbox, .pogo, .unknown,
    ]

    static func space() throws -> DictionaryValueSpace {
        DictionaryValueSpace([
            "player": AnyValueSpace(try BoxValueSpace(low: -1, high: 1, shape: [playerFeatures])),
            "entities": AnyValueSpace(try BoxValueSpace(low: -1, high: 1, shape: [maxEntities, entityFeatures])),
            "entity_mask": AnyValueSpace(try MultiBinaryValueSpace(n: maxEntities)),
        ])
    }

    /// Flattens entities, entity_mask, then player in Gymnazo's sorted dictionary-key order.
    public static func flatten(_ observation: [String: SpaceValue]) throws -> TensorValue<Float> {
        guard observation.count == 3,
            case .floatTensor(let entities) = observation["entities"], entities.shape == [maxEntities, entityFeatures],
            case .boolTensor(let mask) = observation["entity_mask"], mask.shape == [maxEntities],
            case .floatTensor(let player) = observation["player"], player.shape == [playerFeatures]
        else {
            throw GymnazoError.invalidState("Invalid MetroRunner version-5 structured observation.")
        }
        return try TensorValue(entities.elements + mask.elements.map { $0 ? Float(1) : 0 } + player.elements, shape: [flatCount])
    }
}

extension MetroRunnerModel {
    func observation() throws -> [String: SpaceValue] {
        let s = motor, custom = motor.customJump, rules = catalog.rules
        let player: [Double] = [
            s.worldX / 20, s.y / 200, (s.y - surface) / 200, s.vy / 500,
            speed / 440, Double(s.lane - 1), s.laneTo - 1, s.laneFrom - 1,
            s.laneFraction, Double(s.laneDirection), s.grounded ? 1 : 0, surface / 150,
            s.rolling / 0.6, s.sinceLanding / 0.15, s.laneTravel / 20, s.laneDuration / 0.27,
            s.elapsed / 179, power(.magnet) / rules.duration(.magnet, level: config.powerLevel),
            power(.sneakers) / rules.duration(.sneakers, level: config.powerLevel),
            power(.multiplier) / rules.duration(.multiplier, level: config.powerLevel),
            power(.jetpack) / rules.duration(.jetpack, level: config.powerLevel),
            invulnerable / 3, stumble / 5, max(0, 3 - s.elapsed) / 3,
            s.bodyHeight / 9, s.jumpHeight / 40, -1, 1, s.laneElapsed / 0.27,
            s.canJump && power(.jetpack) <= 0 ? 1 : 0, s.diving ? 1 : 0,
            flightStartY / 200, min(1, flightElapsed / rules.jetpack.adjustmentTime),
            custom != nil ? 1 : 0, custom?.source == .pogo ? 1 : 0,
            custom.map { $0.elapsed / $0.duration } ?? 0, (custom?.duration ?? 0) / 4,
            (custom?.startY ?? 0) / 200, (custom?.height ?? 0) / 200,
            (custom?.speed ?? 0) / 440, power(.jetpack) > 0 ? flightSpeed / 440 : 0,
        ]
        let visible = entities.filter { $0.alive && $0.z + $0.depth >= s.z - 4 && $0.z <= s.z + 600 }
        let hazards = visible.filter { $0.kind.isHazard || $0.kind == .ramp }.sorted(by: imminent)
        let pickups = visible.filter { !$0.kind.isHazard && $0.kind != .ramp }.sorted(by: imminent)
        var selected = Array(hazards.prefix(48)) + Array(pickups.prefix(16))
        let remainder = (Array(hazards.dropFirst(48)) + Array(pickups.dropFirst(16))).sorted(by: imminent)
        selected.append(contentsOf: remainder.prefix(MetroRunnerObservation.maxEntities - selected.count))
        selected.sort(by: imminent)
        let features = MetroRunnerObservation.entityFeatures
        let offset = MetroRunnerObservation.kinds.count
        var rows = [Float](repeating: 0, count: MetroRunnerObservation.maxEntities * features)
        var mask = [Bool](repeating: false, count: MetroRunnerObservation.maxEntities)
        for (index, entity) in selected.enumerated() {
            let type = MetroRunnerObservation.kinds.firstIndex(of: entity.kind) ?? offset - 1
            rows[index * features + type] = 1
            let top = entity.y + entity.height
            let support = entity.kind == .ramp ? entity.surface(at: s.z) : (entity.kind.isSupport ? top : 0)
            let values = bounded([
                entity.x / 40, entity.width / 80, (entity.z - s.z) / 2400,
                (entity.z + entity.depth - s.z) / 2400, entity.collisionBottom / 200, entity.collisionTop / 200,
                (entity.velocity - speed) / 1320, support / 150,
                entity.kind == .ramp ? (entity.rampRun > 0 ? entity.rampRun : entity.depth) / 2400 : 0,
                entity.inAir ? 1 : 0,
            ])
            rows.replaceSubrange((index * features + offset)..<(index * features + features), with: values)
            mask[index] = true
        }
        return ["player": .floatTensor(try TensorValue(bounded(player), shape: [MetroRunnerObservation.playerFeatures])),
                "entities": .floatTensor(try TensorValue(rows, shape: [MetroRunnerObservation.maxEntities, features])),
                "entity_mask": .boolTensor(try TensorValue(mask, shape: [MetroRunnerObservation.maxEntities]))]
    }

    private func bounded(_ values: [Double]) -> [Float] {
        values.map { $0.isNaN ? 0 : Float(min(1, max(-1, $0))) }
    }

    private func imminent(_ a: MetroRunnerEntity, _ b: MetroRunnerEntity) -> Bool {
        let timeA = max(0, a.z - motor.z) / max(1, speed - a.velocity)
        let timeB = max(0, b.z - motor.z) / max(1, speed - b.velocity)
        if timeA != timeB { return timeA < timeB }
        if a.z != b.z { return a.z < b.z }
        if a.x != b.x { return a.x < b.x }
        if a.kind != b.kind { return a.kind.rawValue < b.kind.rawValue }
        return a.y < b.y
    }
}
