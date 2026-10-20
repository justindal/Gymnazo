import Foundation
import simd

extension MetroRunnerRoute {
    mutating func instantiate(_ name: String, origin: Double, speed: Double) throws -> ([MetroRunnerEntity], Double) {
        guard let chunk = catalog.chunks[name] else {
            throw GymnazoError.invalidConfiguration("Unknown MetroRunner chunk: \(name).")
        }
        var entities: [MetroRunnerEntity] = []
        Self.visit(chunk.root, position: .zero, rotation: simd_quatd(angle: 0, axis: SIMD3(0, 1, 0)),
                   scale: SIMD3(repeating: 1), force: true, origin: origin, speed: speed,
                   length: chunk.length, generator: &generator, entities: &entities)
        return (entities, chunk.length)
    }

    private static func visit(
        _ node: MetroRunnerCatalog.Node,
        position: SIMD3<Double>, rotation: simd_quatd, scale: SIMD3<Double>, force: Bool = false,
        origin: Double, speed: Double, length: Double,
        generator: inout SeededGenerator, entities: inout [MetroRunnerEntity]
    ) {
        guard node.active || force else { return }
        var world = position + rotation.act(vector(node.position) * scale)
        let q = rotation * simd_quatd(vector: SIMD4(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]))
        var sc = scale * vector(node.scale)
        if let lanes = node.lanes, !lanes.isEmpty {
            world.x = lanes[Int.random(in: lanes.indices, using: &generator)]
        }
        if let probability = node.mirror, Double(generator.uniform(in: 0...1)) < probability { sc.x *= -1 }
        let label = (node.asset.isEmpty ? node.name : node.asset).lowercased()
        let x = world.x, y = world.y, z = world.z

        if abs(x) < 45, y >= -10, y < 150, z >= -30, z <= max(2500, length / 11.25 * 15) {
            if label.contains("train_static") || label.contains("train_moving") {
                let moving = label.contains("moving")
                let cars = number(in: label, pattern: "(?:static|moving)(?:_preportal)?_(\\d+)") ?? 3
                var xx = x, bottom = y + 1.2, front = origin + z + 3
                var width = 17.0, height = 28.0, depth = 60 * cars - 4
                if let collider = node.colliders.first {
                    let center = q.act(vector(collider.center) * sc)
                    let size = absVector(vector(collider.size) * sc)
                    xx = x + center.x
                    bottom = y + center.y - size.y / 2
                    front = origin + z + center.z - size.z / 2
                    width = size.x; height = size.y; depth = size.z
                }
                let factor = moving ? node.movingSpeed : 0
                entities.append(MetroRunnerEntity(kind: moving ? .movingTrain : .train,
                    x: xx, y: bottom, z: front, width: width, depth: depth * (1 + factor), height: height,
                    velocity: -factor * speed, carrierZ: moving ? front : nil, motionFactor: factor))
            } else if label.contains("train_ramp") {
                entities.append(MetroRunnerEntity(kind: .ramp, x: x, y: y, z: origin + z - 49,
                    width: 17, depth: 80, height: 29.2, rampRun: 62))
            } else if (node.name.lowercased().contains("blocker") || label.contains("obstacle_barrier")) && node.placer {
                let kind: MetroRunnerEntityKind = label.contains("jump") ? .jump : .roll
                entities.append(MetroRunnerEntity(kind: kind, x: x, y: y, z: origin + z,
                    depth: 8, height: kind == .jump ? 12 : 24))
            } else if label.hasPrefix("special_gate") {
                let lanes: [Double]
                switch label.split(separator: "_").last {
                case "left": lanes = [-20]
                case "right": lanes = [20]
                case "mid": lanes = [0]
                case "sides": lanes = [-20, 20]
                default: lanes = []
                }
                for lane in lanes {
                    entities.append(MetroRunnerEntity(kind: .gate, x: (sc.x < 0 ? -1 : 1) * lane,
                        y: 0, z: origin + z, width: 17, depth: 25, height: 60))
                }
            }
            if ["Station_Platform", "Platform", "collider_R"].contains(node.name) {
                for collider in node.colliders {
                    let center = q.act(vector(collider.center) * sc)
                    let size = absVector(vector(collider.size) * sc)
                    entities.append(MetroRunnerEntity(kind: .platform, x: x + center.x,
                        y: y + center.y - size.y / 2, z: origin + z + center.z - size.z / 2,
                        width: size.x, depth: size.z, height: size.y))
                }
            }
            if node.spawn {
                let name = node.name.lowercased()
                if name.contains("coin") || node.coin {
                    let span = number(in: name, pattern: "over_(\\d+)") ?? Double(max(0, (node.coinCount - 1) * 30))
                    for index in 0..<node.coinCount {
                        let t = Double(index) / Double(max(1, node.coinCount - 1))
                        let lift = name.contains("curve") ? sin(.pi * t) * 15 : 0
                        entities.append(MetroRunnerEntity(kind: .coin, x: x, y: y + lift,
                            z: origin + z + t * span, width: 3, depth: 3, height: 3))
                        _ = generator.next()
                    }
                } else if name.contains("pickup"), Double(generator.uniform(in: 0...1)) < (name.contains("special") ? 0.34 : 0.48) {
                    let kinds: [MetroRunnerEntityKind] = [.magnet, .sneakers, .multiplier, .jetpack, .mystery, .key]
                    let index = generator.weightedIndex([28, 25, 25, 12, 8, 2]) ?? 0
                    entities.append(MetroRunnerEntity(kind: kinds[index], x: x, y: y,
                        z: origin + z, width: 5, depth: 5, height: 5))
                    _ = generator.next()
                }
            }
        }

        var children = node.children
        if let probability = node.randomChild {
            let labels = children.map { $0.asset.lowercased() }
            if labels.contains(where: { $0.contains("barrier_jump") }), labels.contains(where: { $0.contains("barrier_roll") }) {
                children.removeAll { $0.asset.lowercased().contains("barrier_standard") }
            }
            if !children.isEmpty, Double(generator.uniform(in: 0...1)) < probability {
                let child = children[Int.random(in: children.indices, using: &generator)]
                visit(child, position: world, rotation: q, scale: sc, force: true,
                      origin: origin, speed: speed, length: length, generator: &generator, entities: &entities)
            }
        } else {
            for child in children {
                visit(child, position: world, rotation: q, scale: sc,
                      origin: origin, speed: speed, length: length, generator: &generator, entities: &entities)
            }
        }
    }

    private static func vector(_ values: [Double]) -> SIMD3<Double> { SIMD3(values[0], values[1], values[2]) }
    private static func absVector(_ v: SIMD3<Double>) -> SIMD3<Double> { SIMD3(abs(v.x), abs(v.y), abs(v.z)) }
    private static func number(in text: String, pattern: String) -> Double? {
        guard let range = text.range(of: pattern, options: .regularExpression) else { return nil }
        return Double(text[range].split(separator: "_").last ?? "")
    }
}
