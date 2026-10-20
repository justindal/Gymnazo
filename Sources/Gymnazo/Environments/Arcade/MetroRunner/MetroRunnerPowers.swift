import Foundation

extension MetroRunnerModel {
    mutating func updateCoinCurves() {
        for index in entities.indices {
            let item = entities[index]
            guard !item.curveSpawner.isEmpty, let origin = item.spawnOrigin else { continue }
            let offset = catalog.rules.coinPosition(item.curveSpawner, t: item.curveT, speedModifier: motor.speed / 110)
            entities[index].x = origin.x + offset.x
            entities[index].y = origin.y + offset.y
            entities[index].z = origin.z + offset.z
        }
    }

    func flyoverSuppresses(_ item: MetroRunnerEntity) -> Bool {
        guard !item.source.isEmpty, !item.kind.isHazard, item.kind != .ramp else { return false }
        let anchor = item.spawnOrigin?.z ?? item.z
        return item.disableOnFlyover || (anchor >= motor.z + item.preserveFlyoverDistance && anchor < flightEndZ)
    }

    mutating func collect(_ index: Int) {
        let item = entities[index]
        entities[index].alive = false
        event(item.kind.rawValue, x: item.x, y: item.y, z: item.z)
        switch item.kind {
        case .coin: coins += 1
        case .key: keys += 1
        case .mystery: mysteryBoxes += 1
        case .pogo: startPogo()
        default:
            if item.kind.isPower {
                powers[item.kind] = catalog.rules.duration(item.kind, level: config.powerLevel)
                if item.kind == .sneakers { stumble = 0 }
                if item.kind == .jetpack { startFlight() }
            }
        }
    }

    private mutating func startFlight() {
        let rules = catalog.rules.jetpack
        motor.customJump = nil
        motor.grounded = false
        flightStartY = motor.y
        flightElapsed = 0
        flightSpeed = motor.speed * rules.speedFactor
        flightEndZ = motor.z + power(.jetpack) * flightSpeed
        motor.forwardSpeedOverride = flightSpeed
        for index in entities.indices where flyoverSuppresses(entities[index]) { entities[index].alive = false }
        var z = motor.z + flightSpeed * rules.adjustmentTime + rules.coinOffset
        var lane = 0, remaining = rules.coinsPerNode
        while z < flightEndZ {
            entities.append(MetroRunnerEntity(kind: .coin, x: Double(lane) * 20, y: rules.height + rules.coinHeight,
                z: z, width: 3, depth: 3, height: 3))
            remaining -= 1
            if remaining == 0 {
                let delta = Int.random(in: -1...1, using: &gameplay)
                let nextLane = max(-1, min(1, lane + delta))
                if nextLane != lane {
                    z += rules.coinSpacing
                    entities.append(MetroRunnerEntity(kind: .coin, x: (Double(lane) + Double(delta) * 0.5) * 20,
                        y: rules.height + rules.coinHeight, z: z, width: 3, depth: 3, height: 3))
                }
                lane = nextLane
                remaining = rules.coinsPerNode
            }
            z += rules.coinSpacing
        }
    }

    mutating func startPogo() {
        let rules = catalog.rules.pogo
        let originY = motor.y, originZ = motor.z
        motor.startCustomJump(height: max(0, rules.height - originY), curve: rules.curve,
            distance: rules.distance, speed: motor.speed, source: .pogo, disallowJump: true, fastDive: true)
        let end = rules.distance - rules.coinEnd
        for index in 0..<rules.rows {
            let offset = rules.coinStart + (end - rules.coinStart) * Double(index) / Double(rules.rows)
            let height = originY + rules.coinHeight + rules.height * rules.curve.value(at: offset / rules.distance)
            for lane in [-20.0, 0, 20] {
                entities.append(MetroRunnerEntity(kind: .coin, x: lane, y: height, z: originZ + offset,
                    width: 3, depth: 3, height: 3, inAir: true))
            }
        }
        if let kind = spawnRules.choosePogo() {
            entities.append(MetroRunnerEntity(kind: kind, x: motor.worldX, y: originY + rules.rewardHeight + rules.height,
                z: originZ + rules.distance + rules.rewardOffset, width: 5, depth: 5, height: 5))
        }
    }
}
