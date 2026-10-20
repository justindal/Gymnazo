import Foundation

extension MetroRunnerModel {
    mutating func tick(_ command: MetroRunnerAction) throws {
        let dt = MetroRunnerMotor.dt
        let oldZ = motor.z, oldY = motor.y
        let wasFlying = power(.jetpack) > 0
        for key in Array(powers.keys) { powers[key] = max(0, power(key) - dt) }
        invulnerable = max(0, invulnerable - dt)
        stumble = max(0, stumble - dt)
        let flying = power(.jetpack) > 0
        motor.forwardSpeedOverride = flying ? flightSpeed : nil
        motor.laneChangeDuration = flying ? catalog.rules.jetpack.laneDuration : 0.27
        motor.jumpHeight = power(.sneakers) > 0 ? 40 : 20
        motor.jumpCurve = power(.sneakers) > 0 ? catalog.rules.sneakersCurve : nil
        var action = command
        if flying && (action == .jump || action == .down) { action = .none }
        if action == .jump && motor.canJump {
            jumps += 1
            motor.rolling = 0
            event("jump")
        }
        for index in entities.indices where entities[index].carrierZ == nil {
            entities[index].z += entities[index].velocity * dt
        }
        let futureZ = motor.z + speed * dt
        updateMovingTrains(at: futureZ)
        updateCoinCurves()
        if action == .left || action == .right {
            motor.changeLane(action == .left ? -1 : 1)
            action = .none
        }
        let futureX: Double
        if motor.laneElapsed < motor.laneDuration {
            let fraction = min(1, (motor.laneElapsed + dt) / motor.laneDuration)
            futureX = (motor.laneFrom + (motor.laneTo - motor.laneFrom) * fraction - 1) * 20
        } else {
            futureX = motor.worldX
        }
        var support = 0.0
        for entity in entities where entity.alive && abs(entity.x - futureX) <= entity.width / 2 + 1 {
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
        if flying {
            flightElapsed += dt
            let fraction = min(1, flightElapsed / catalog.rules.jetpack.adjustmentTime)
            let blend = fraction * fraction * (3 - 2 * fraction)
            motor.y = flightStartY + (catalog.rules.jetpack.height - flightStartY) * blend
            motor.vy = (motor.y - oldY) / dt
            motor.grounded = false; motor.rolling = 0; motor.diving = false
        } else if wasFlying {
            invulnerable = max(invulnerable, 2)
            motor.grounded = false
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
        let bodyHeight = motor.bodyHeight
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
                    motor.y = top
                    motor.vy = 0
                    motor.grounded = true
                    surface = top
                }
                if motor.y >= top - 0.5 || motor.y + bodyHeight < entity.y { continue }
            } else if entity.kind == .roll {
                if motor.y + bodyHeight < entity.collisionBottom || motor.y > entity.collisionTop { continue }
            } else if motor.y >= entity.y + entity.height { continue }
            if entity.kind == .bush || entity.kind == .powerbox {
                if !entity.contacted {
                    entities[index].contacted = true
                    hit(entity, side: true)
                }
            } else {
                hit(entity, side: dx > entity.width / 2 - 1)
            }
            if dead { break }
        }
    }

    mutating func hit(_ obstacle: MetroRunnerEntity, side: Bool) {
        guard invulnerable <= 0 else { return }
        if side && stumble <= 0 {
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
