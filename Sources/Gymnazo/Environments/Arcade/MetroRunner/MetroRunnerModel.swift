import Foundation

struct MetroRunnerModel: Sendable {
    let config: MetroRunnerConfig
    let catalog: MetroRunnerCatalog
    var curriculum: MetroRunnerCurriculum
    var motor = MetroRunnerMotor()
    var route: MetroRunnerRoute?
    var gameplay = SeededGenerator(seed: 0)
    var seedGenerator = SeededGenerator(seed: UInt64.random(in: .min ... .max))
    var routeSeed: UInt64 = 0
    var entities: [MetroRunnerEntity] = []
    var powers: [MetroRunnerEntityKind: Double] = [:]
    var board = 0.0
    var coins = 0
    var keys = 0
    var score = 0
    var boards = 3
    var jumps = 0
    var travelScore = 0.0
    var dead = false
    var deathReason = ""
    var invulnerable = 0.0
    var stumble = 0.0
    var surface = 0.0
    var nextChunk = 0.0
    var physicsSteps = 0
    var needsReset = true
    var closed = false
    var events: [SpaceValue] = []
    var missions = [false, false, false]

    init(config: MetroRunnerConfig) throws {
        try config.validate()
        self.config = config
        self.curriculum = config.curriculum
        self.catalog = try MetroRunnerCatalog.bundled.get()
        if let section = config.section, catalog.sections[section] == nil {
            throw GymnazoError.invalidConfiguration("Unknown MetroRunner section: \(section).")
        }
    }

    mutating func reset(seed: UInt64?, options: ValueResetOptions?) throws -> ValueReset<[String: SpaceValue]> {
        try ensureOpen()
        let supported: Set<String> = ["curriculum", "route_seed"]
        guard Set((options ?? [:]).keys).isSubset(of: supported) else {
            throw GymnazoError.invalidConfiguration("MetroRunner reset accepts curriculum and route_seed.")
        }
        var selectedCurriculum = curriculum
        if let value = options?["curriculum"] {
            guard case .text(let name) = value, let parsed = MetroRunnerCurriculum(rawValue: name) else {
                throw GymnazoError.invalidConfiguration("Invalid MetroRunner curriculum.")
            }
            selectedCurriculum = parsed
        }
        var selectedSeed: UInt64?
        if let value = options?["route_seed"] {
            switch value {
            case .discrete(let number) where number >= 0: selectedSeed = UInt64(number)
            case .text(let text): selectedSeed = UInt64(text)
            default: break
            }
            guard selectedSeed != nil else {
                throw GymnazoError.invalidConfiguration("MetroRunner route_seed must be a nonnegative integer or UInt64 decimal string.")
            }
        }
        if let seed { seedGenerator = SeededGenerator(seed: seed) }
        routeSeed = selectedSeed ?? seedGenerator.next()
        route = try MetroRunnerRoute(catalog: catalog, seed: routeSeed, section: config.section)
        gameplay = SeededGenerator(seed: routeSeed ^ 0x5EEDC01)
        curriculum = selectedCurriculum
        motor = MetroRunnerMotor()
        entities = []
        powers = [:]
        board = 0
        coins = 0; keys = 0; score = 0; boards = 3; jumps = 0
        travelScore = 0
        dead = false; deathReason = ""
        invulnerable = 0; stumble = 0; surface = 0; nextChunk = 0
        physicsSteps = 0
        events = []; missions = [false, false, false]
        try ensureRoute()
        needsReset = false
        return ValueReset(observation: try observation(), info: info)
    }

    mutating func step(_ action: Int) throws -> ValueStep<[String: SpaceValue]> {
        try ensureOpen()
        guard !needsReset else { throw GymnazoError.stepBeforeReset }
        guard let command = MetroRunnerAction(rawValue: action) else {
            throw GymnazoError.invalidAction("MetroRunner actions must be integers from 0 through 5.")
        }
        let oldZ = motor.z, oldCoins = coins
        let limit = max(1, Int(ceil(config.maxEpisodeSeconds * 60)))
        events = []
        var ticks = 0
        for index in 0..<config.frameSkip {
            try tick(index == 0 ? command : .none)
            physicsSteps += 1
            ticks += 1
            if dead || physicsSteps >= limit { break }
        }
        let truncated = !dead && physicsSteps >= limit
        needsReset = dead || truncated
        let distanceReward = (motor.z - oldZ) * config.reward.distanceScale
        let coinReward = Double(coins - oldCoins) * config.reward.coinWeight
        let deathReward = dead ? -config.reward.deathPenalty : 0
        var details = info
        details["events"] = .sequence(events)
        details["action_physics_steps"] = .discrete(ticks)
        details["reward_components"] = .dictionary([
            "distance": .double(distanceReward), "coins": .double(coinReward), "death": .double(deathReward),
        ])
        if truncated { details[EnvInfoKey.timeLimitTruncated] = .bool(true) }
        return ValueStep(observation: try observation(), reward: distanceReward + coinReward + deathReward,
                         terminated: dead, truncated: truncated, info: details)
    }

    mutating func ensureRoute() throws {
        guard var route else { throw GymnazoError.invalidState("MetroRunner route is not initialized.") }
        while nextChunk < motor.z + 2400 {
            let name = try route.next(at: nextChunk)
            let (items, length) = try route.instantiate(name, origin: nextChunk, speed: motor.speed)
            entities.append(contentsOf: items)
            nextChunk += length
            route.placed(at: nextChunk)
        }
        self.route = route
        entities = entities.filter { $0.alive && $0.z + $0.depth > motor.z - 150 && includes($0) }
        updateMovingTrains(at: motor.z)
    }

    private func includes(_ entity: MetroRunnerEntity) -> Bool {
        switch curriculum {
        case .barriers: [.jump, .roll, .coin].contains(entity.kind) && entity.y < 15
        case .static: [.train, .ramp, .jump, .roll, .gate, .platform, .coin].contains(entity.kind)
        case .moving: entity.kind.isHazard || entity.kind == .ramp || entity.kind == .coin
        case .full: true
        }
    }

    mutating func updateMovingTrains(at z: Double) {
        for index in entities.indices {
            if let carrier = entities[index].carrierZ {
                let factor = entities[index].motionFactor
                entities[index].z = (1 + factor) * carrier - factor * z
                entities[index].velocity = -factor * motor.speed
            }
        }
    }

    var info: ValueInfo {
        ["route_seed": .text(String(routeSeed)), "distance": .double(motor.z),
         "coins": .discrete(coins), "score": .discrete(score), "death_reason": .text(deathReason),
         "elapsed_seconds": .double(Double(physicsSteps) / 60), "physics_steps": .discrete(physicsSteps),
         "curriculum": .text(curriculum.rawValue), "observation_version": .discrete(2)]
    }

    var snapshot: MetroRunnerSnapshot {
        MetroRunnerSnapshot(x: motor.worldX, y: motor.y, surface: surface, distance: motor.z, speed: motor.speed,
                            grounded: motor.grounded, rolling: motor.rolling > 0, coins: coins, score: score,
                            boards: boards, boardRemaining: board, powers: powers, dead: dead,
                            entities: entities.filter(\.alive).map(\.snapshot))
    }

    mutating func event(_ name: String, x: Double? = nil, y: Double? = nil, z: Double? = nil) {
        events.append(.tuple([.text(name), .double(x ?? motor.worldX), .double(y ?? motor.y), .double(z ?? motor.z)]))
    }

    private func ensureOpen() throws {
        guard !closed else { throw GymnazoError.invalidState("MetroRunner is closed.") }
    }
}
