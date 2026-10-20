import Foundation

extension MetroRunnerModel {
    mutating func tick(_ command: MetroRunnerAction) throws {
        let dt = MetroRunnerMotor.dt
        let oldZ = motor.z, oldY = motor.y
        let wasFlying = power(.jetpack) > 0
        for key in Array(powers.keys) { powers[key] = max(0, power(key) - dt) }
        board = max(0, board - dt)
        invulnerable = max(0, invulnerable - dt)
        stumble = max(0, stumble - dt)
        motor.jumpHeight = power(.sneakers) > 0 ? 40 : 20
        var action = command
        if action == .board {
            if boards > 0, board <= 0, power(.jetpack) <= 0 {
                boards -= 1
                board = 30
                event("board")
            }
            action = .none
        }
        if action == .jump, motor.grounded {
            jumps += 1
            motor.rolling = 0
            event("jump")
        }
        for index in entities.indices where entities[index].carrierZ == nil {
            entities[index].z += entities[index].velocity * dt
        }
        let futureZ = motor.z + motor.speed * dt
        updateMovingTrains(at: futureZ)
        var support = 0.0
        for entity in entities where entity.alive && abs(entity.x - motor.worldX) <= entity.width / 2 + 1 {
            guard entity.z - 2 <= futureZ, futureZ <= entity.z + entity.depth else { continue }
            let top = entity.surface(at: futureZ)
            if entity.kind == .ramp {
                if motor.y >= top - 4 { support = max(support, top) }
            } else if entity.kind.isSupport {
                let tolerance = motor.grounded && surface > 0 ? 4.0 : 1.0
                if motor.y >= top - tolerance { support = max(support, top) }
            }
        }
        if motor.grounded {
            if support < surface - 1 {
                motor.grounded = false
                motor.vy = 0
            } else {
                motor.y = max(motor.y, support)
            }
        }
        surface = support
        motor.ground = support
        motor.step(action)
        let flying = power(.jetpack) > 0
        if flying {
            motor.y += (65 - motor.y) * min(1, dt * 8)
            motor.vy = 0; motor.grounded = false; motor.rolling = 0
        } else if wasFlying {
            invulnerable = max(invulnerable, 2)
            motor.vy = 0; motor.grounded = false
        }
        collide(oldZ: oldZ, oldY: oldY, flying: flying)
        travelScore += (motor.z - oldZ) / 10 * (power(.multiplier) > 0 ? 2 : 1)
        score = Int(travelScore) + coins * 5
        let newMissions = [coins >= 30, motor.z >= 1500, jumps >= 5]
        for index in newMissions.indices where newMissions[index] && !missions[index] { event("mission") }
        missions = newMissions
        try ensureRoute()
    }

    func power(_ kind: MetroRunnerEntityKind) -> Double { powers[kind] ?? 0 }

    private mutating func collide(oldZ: Double, oldY: Double, flying: Bool) {
        let bodyHeight = motor.rolling > 0 ? 4.0 : 9.0
        let count = entities.count
        for index in 0..<count {
            let entity = entities[index]
            guard entity.alive else { continue }
            if !entity.kind.isHazard && entity.kind != .ramp {
                let dx = abs(entity.x - motor.worldX), dz = entity.z - motor.z
                let dy = abs(entity.y - (motor.y + 5))
                if entity.kind == .coin, power(.magnet) > 0, dz > -15, dz < 90, dy < 40 {
                    collect(index)
                } else if dx < 6, dy < 9, oldZ - 4 <= entity.z, entity.z <= motor.z + 5 {
                    collect(index)
                }
                continue
            }
            if entity.kind == .ramp || flying { continue }
            if motor.z + 2 < entity.z || oldZ - 2 > entity.z + entity.depth { continue }
            let dx = abs(entity.x - motor.worldX)
            if dx >= entity.width / 2 + 2.6 { continue }
            if entity.kind.isSupport {
                let top = entity.y + entity.height
                if motor.vy <= 0, oldY >= top, motor.y <= top {
                    motor.y = top; motor.vy = 0; motor.grounded = true; surface = top
                }
                if motor.y >= top - 0.5 || motor.y + bodyHeight < entity.y { continue }
            } else if entity.kind == .roll {
                if motor.y + bodyHeight < entity.y + 5 || motor.y > entity.y + 24 { continue }
            } else if motor.y >= entity.y + entity.height { continue }
            hit(entity, side: dx > entity.width / 2 - 1)
            if dead { break }
        }
    }

    mutating func collect(_ index: Int) {
        let item = entities[index]
        entities[index].alive = false
        event(item.kind.rawValue, x: item.x, y: item.y, z: item.z)
        switch item.kind {
        case .coin: coins += 1
        case .key: keys += 1
        case .mystery:
            coins += Int.random(in: 20...60, using: &gameplay)
            boards += 1
        default:
            if let duration = item.kind.powerDuration {
                powers[item.kind] = duration
                if item.kind == .jetpack {
                    motor.grounded = false
                    var lane = 0.0
                    for index in 0..<48 {
                        if index % 8 == 0 { lane = Double(Int.random(in: -1...1, using: &gameplay)) * 20 }
                        entities.append(MetroRunnerEntity(kind: .coin, x: lane, y: 71,
                            z: motor.z + 60 + Double(index) * 24, width: 3, depth: 3, height: 3))
                    }
                }
            }
        }
    }

    mutating func hit(_ obstacle: MetroRunnerEntity, side: Bool) {
        guard invulnerable <= 0 else { return }
        if board > 0 {
            board = 0
            invulnerable = 2
            for index in entities.indices where entities[index].kind.isHazard && abs(entities[index].z - motor.z) < 150 {
                entities[index].alive = false
            }
            event("save")
        } else if side && stumble <= 0 {
            stumble = 5
            invulnerable = 0.8
            event("stumble")
        } else {
            dead = true
            switch obstacle.kind {
            case .roll: deathReason = "Roll under the overhead barrier"
            case .jump: deathReason = "Jump over the low barrier"
            case .gate: deathReason = "Choose an open tunnel lane"
            case .train: deathReason = "Train collision"
            case .movingTrain: deathReason = "Oncoming train collision"
            default: deathReason = "Collision"
            }
            event("crash")
        }
    }
}
