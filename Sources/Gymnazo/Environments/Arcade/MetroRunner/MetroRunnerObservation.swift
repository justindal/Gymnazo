import Foundation

/// The structured observation contract, independent of route identifiers and future random choices.
public enum MetroRunnerObservation {
    public static let version = 2
    public static let maxEntities = 64
    public static let entityFeatures = 22
    public static let playerFeatures = 32
    public static let flatCount = 1504

    static func space() throws -> DictionaryValueSpace {
        DictionaryValueSpace([
            "player": AnyValueSpace(try BoxValueSpace(low: -1, high: 1, shape: [32])),
            "entities": AnyValueSpace(try BoxValueSpace(low: -1, high: 1, shape: [64, 22])),
            "entity_mask": AnyValueSpace(try MultiBinaryValueSpace(n: 64)),
        ])
    }

    /// Flattens entities, entity_mask, then player in Gymnazo's sorted dictionary-key order.
    public static func flatten(_ observation: [String: SpaceValue]) throws -> TensorValue<Float> {
        guard observation.count == 3,
            case .floatTensor(let entities) = observation["entities"], entities.shape == [64, 22],
            case .boolTensor(let mask) = observation["entity_mask"], mask.shape == [64],
            case .floatTensor(let player) = observation["player"], player.shape == [32]
        else {
            throw GymnazoError.invalidState("Invalid MetroRunner structured observation.")
        }
        return try TensorValue(entities.elements + mask.elements.map { $0 ? Float(1) : 0 } + player.elements, shape: [flatCount])
    }
}

extension MetroRunnerModel {
    func observation() throws -> [String: SpaceValue] {
        let s = motor
        let player: [Double] = [
            s.worldX / 20, s.y / 150, (s.y - surface) / 150, s.vy / 200,
            s.speed / 220, Double(s.lane - 1), s.laneTo - 1, s.laneFrom - 1,
            s.laneFraction, Double(s.laneDirection), s.grounded ? 1 : 0, surface / 150,
            s.rolling / 0.6, s.sinceLanding / 0.15, s.laneTravel / 20, s.laneDuration / 0.27,
            s.elapsed / 179, power(.magnet) / 15, power(.sneakers) / 15,
            power(.multiplier) / 15, power(.jetpack) / 12, board / 30,
            Double(boards) / 10, invulnerable / 3, stumble / 5, max(0, 3 - s.elapsed) / 3,
            (s.rolling > 0 ? 4.0 : 9) / 9, s.jumpHeight / 40, -1, 1, s.laneElapsed / 0.27,
            boards > 0 && board <= 0 && power(.jetpack) <= 0 ? 1 : 0,
        ]
        let visible = entities.filter { $0.alive && $0.z + $0.depth >= s.z - 4 && $0.z <= s.z + 600 }
        let hazards = visible.filter { $0.kind.isHazard || $0.kind == .ramp }.sorted(by: imminent)
        let pickups = visible.filter { !$0.kind.isHazard && $0.kind != .ramp }.sorted(by: imminent)
        var selected = Array(hazards.prefix(48)) + Array(pickups.prefix(16))
        let remainder = (Array(hazards.dropFirst(48)) + Array(pickups.dropFirst(16))).sorted(by: imminent)
        selected.append(contentsOf: remainder.prefix(64 - selected.count))
        selected.sort(by: imminent)
        var rows = [Float](repeating: 0, count: 64 * 22)
        var mask = [Bool](repeating: false, count: 64)
        for (index, entity) in selected.enumerated() {
            let type = MetroRunnerEntityKind.allCases.firstIndex(of: entity.kind) ?? 13
            rows[index * 22 + type] = 1
            let top = entity.y + entity.height
            let support = entity.kind == .ramp ? entity.surface(at: s.z) : (entity.kind.isSupport ? top : 0)
            let values = bounded([
                entity.x / 40, entity.width / 80, (entity.z - s.z) / 2400,
                (entity.z + entity.depth - s.z) / 2400, entity.y / 150, top / 150,
                (entity.velocity - s.speed) / 440, support / 150,
            ])
            rows.replaceSubrange((index * 22 + 14)..<(index * 22 + 22), with: values)
            mask[index] = true
        }
        return ["player": .floatTensor(try TensorValue(bounded(player), shape: [32])),
                "entities": .floatTensor(try TensorValue(rows, shape: [64, 22])),
                "entity_mask": .boolTensor(try TensorValue(mask, shape: [64]))]
    }

    private func bounded(_ values: [Double]) -> [Float] {
        values.map { $0.isNaN ? 0 : Float(min(1, max(-1, $0))) }
    }

    private func imminent(_ a: MetroRunnerEntity, _ b: MetroRunnerEntity) -> Bool {
        let timeA = max(0, a.z - motor.z) / max(1, motor.speed - a.velocity)
        let timeB = max(0, b.z - motor.z) / max(1, motor.speed - b.velocity)
        if timeA != timeB { return timeA < timeB }
        if a.z != b.z { return a.z < b.z }
        if a.x != b.x { return a.x < b.x }
        if a.kind != b.kind { return a.kind.rawValue < b.kind.rawValue }
        return a.y < b.y
    }
}
