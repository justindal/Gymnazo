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
    var rolling = 0.0
    var grounded = true
    var sinceLanding = 0.0
    var ground = 0.0
    var jumpHeight = 20.0

    var speed: Double { 110 + 110 * min(max(elapsed / 179, 0), 1) }
    var worldX: Double { (x - 1) * 20 }
    var laneFraction: Double { laneDuration <= 0 ? 1 : min(laneElapsed / laneDuration, 1) }

    mutating func step(_ action: MetroRunnerAction) {
        let dt = Self.dt
        switch action {
        case .left: changeLane(-1)
        case .right: changeLane(1)
        case .jump:
            if grounded || sinceLanding <= 0.15 {
                vy = sqrt(400 * jumpHeight)
                grounded = false
                sinceLanding = .infinity
            }
        case .down:
            rolling = 0.6
            if !grounded { vy = -90 }
        case .none, .board: break
        }

        if laneElapsed < laneDuration {
            laneElapsed += dt
            laneTravel += dt * abs(speed)
            x = laneFrom + (laneTo - laneFrom) * min(laneElapsed / laneDuration, 1)
        } else {
            x = Double(lane)
        }
        if !grounded {
            vy -= 200 * dt
            y += vy * dt
            if y <= ground {
                y = ground
                vy = 0
                grounded = true
                sinceLanding = 0
            }
        } else if sinceLanding.isFinite {
            sinceLanding += dt
        }
        rolling = max(0, rolling - dt)
        z += speed * dt
        elapsed += dt
    }

    private mutating func changeLane(_ direction: Int) {
        if laneElapsed < laneDuration {
            guard direction != laneDirection, laneTravel >= 1.5 else { return }
        }
        let target = lane + direction
        guard (0..<3).contains(target) else { return }
        laneFrom = x
        laneTo = Double(target)
        lane = target
        laneElapsed = 0
        laneDuration = 0.27 / max(speed / 110, 1)
        laneDirection = direction
        laneTravel = 0
    }
}
