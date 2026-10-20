import Foundation

struct MetroRunnerMotor: Sendable {
    static let dt = 1.0 / 60
    var lane = 1
    var x = 1.0
    var y = 0.0
    var vy = 0.0
    var z = 0.0
    var elapsed = 0.0
    var laneFrom = 1.0
    var laneTo = 1.0
    var laneElapsed = 0.0
    var laneDuration = 0.0
    var laneDirection = 0
    var laneTravel = 0.0
    var laneChangeDuration = 0.27
    var rolling = 0.0
    var grounded = true
    var diving = false
    var sinceLanding = 0.0
    var ground = 0.0
    var jumpHeight = 20.0
    var forwardSpeedOverride: Double?
    var jumpCurve: MetroRunnerCurve?
    var customJump: MetroRunnerCustomJump?

    var speed: Double { 110 + 110 * min(max(elapsed / 179, 0), 1) }
    var forwardSpeed: Double { forwardSpeedOverride ?? customJump?.speed ?? speed }
    var worldX: Double { (x - 1) * 20 }
    var laneFraction: Double { laneDuration <= 0 ? 1 : min(laneElapsed / laneDuration, 1) }
    var canJump: Bool { customJump?.disallowJump != true && (grounded || sinceLanding < 0.15) }
    var bodyHeight: Double { rolling > 0 ? 4.5 : 9 }

    mutating func step(_ action: MetroRunnerAction) {
        let dt = Self.dt
        if grounded { sinceLanding = 0 }
        switch action {
        case .left: changeLane(-1)
        case .right: changeLane(1)
        case .jump:
            if canJump { startJump() }
        case .down:
            if grounded {
                if rolling <= 0 { rolling = 0.6 }
            } else if !diving {
                vy = customJump?.fastDive == true ? -250 : -90
                customJump = nil
                diving = true
            }
        case .none: break
        }

        if laneElapsed < laneDuration {
            laneElapsed += dt
            laneTravel += dt * abs(forwardSpeed)
            x = laneFrom + (laneTo - laneFrom) * min(laneElapsed / laneDuration, 1)
        } else {
            x = Double(lane)
        }
        if var custom = customJump {
            custom.elapsed += dt
            if custom.elapsed < custom.duration {
                let target = custom.startY + min(custom.height, custom.curve.value(at: custom.elapsed / custom.duration) * custom.height)
                vy = (target - y) / dt
                y = target
                customJump = custom
            } else {
                customJump = nil
            }
        }
        if !grounded && customJump == nil {
            vy -= 200 * dt
            y += vy * dt
            if y <= ground { land(at: ground) }
        }
        if grounded {
            sinceLanding = 0
        } else if sinceLanding.isFinite {
            sinceLanding += dt
        }
        rolling = max(0, rolling - dt)
        if let custom = customJump {
            z = custom.startZ + custom.elapsed * custom.speed
        } else {
            z += forwardSpeed * dt
        }
        elapsed += dt
    }

    mutating func land(at height: Double) {
        y = height
        vy = 0
        grounded = true
        sinceLanding = 0
        if diving { rolling = 0.6 }
        diving = false
        customJump = nil
    }

    mutating func startCustomJump(height: Double, curve: MetroRunnerCurve, distance: Double, speed: Double,
                                  source: MetroRunnerEntityKind, disallowJump: Bool = false, fastDive: Bool = false) {
        customJump = MetroRunnerCustomJump(curve: curve, startY: y, startZ: z, height: height,
            speed: speed, duration: distance / speed, source: source, disallowJump: disallowJump, fastDive: fastDive)
        grounded = false
        diving = false
        rolling = 0
        sinceLanding = .infinity
    }

    private mutating func startJump() {
        if let jumpCurve {
            startCustomJump(height: jumpHeight, curve: jumpCurve, distance: sqrt(2 * jumpHeight / 200) * speed,
                            speed: speed, source: .sneakers)
        } else {
            vy = sqrt(400 * jumpHeight)
            grounded = false
            diving = false
            sinceLanding = .infinity
        }
    }

    mutating func changeLane(_ direction: Int) {
        let reversing = laneElapsed < laneDuration
        if reversing {
            guard direction != laneDirection, laneTravel >= 1.5 else { return }
        }
        let target = lane + direction
        guard (0..<3).contains(target) else { return }
        laneFrom = x
        laneTo = Double(target)
        lane = target
        laneElapsed = 0
        laneDuration = laneChangeDuration / max(speed / 110, 1)
        if reversing { laneDuration *= abs(Double(target) - x) }
        laneDirection = direction
        laneTravel = 0
    }
}
