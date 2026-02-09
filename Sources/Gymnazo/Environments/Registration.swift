//
// Registration.swift
//

public struct EnvOptions: Sendable, ExpressibleByDictionaryLiteral {
    public var storage: [String: any Sendable]

    public init(_ storage: [String: any Sendable] = [:]) {
        self.storage = storage
    }

    public init(dictionaryLiteral elements: (String, any Sendable)...) {
        var values: [String: any Sendable] = [:]
        values.reserveCapacity(elements.count)
        for (key, value) in elements {
            values[key] = value
        }
        self.storage = values
    }

    public subscript(_ key: String) -> (any Sendable)? {
        get { storage[key] }
        set { storage[key] = newValue }
    }

    public var isEmpty: Bool { storage.isEmpty }
    public var count: Int { storage.count }
}

public struct WrapperSpec: Sendable {
    public let id: String
    public let entryPoint: @Sendable (any Env, EnvOptions) throws -> any Env
    public var options: EnvOptions = [:]

    public init(
        id: String,
        entryPoint: @escaping @Sendable (any Env, EnvOptions) throws -> any Env,
        options: EnvOptions = [:]
    ) {
        self.id = id
        self.entryPoint = entryPoint
        self.options = options
    }
}

public struct EnvSpec: Sendable {
    public typealias EntryPoint = @Sendable (EnvOptions) throws -> any Env

    public let id: String
    public var entryPoint: EntryPoint

    public var displayName: String?
    public var description: String?
    public var category: EnvCategory?

    public var rewardThreshold: Double? = nil
    public var nondeterministic: Bool = false

    public var maxEpisodeSteps: Int? = nil
    public var orderEnforce: Bool = true
    public var disableEnvChecker: Bool = false

    public var options: EnvOptions = [:]

    public var additionalWrappers: [WrapperSpec] = []

    public var namespace: String? {
        if let slashRange = id.range(of: "/") {
            return String(id[..<slashRange.lowerBound])
        }
        return nil
    }

    public var name: String {
        var namePart = id
        if let slashRange = id.range(of: "/") {
            namePart = String(id[slashRange.upperBound...])
        }
        if let dashRange = namePart.range(of: "-") {
            namePart = String(namePart[..<dashRange.lowerBound])
        }
        return namePart
    }

    public var version: Int? {
        if let versionString = id.split(separator: "-").last,
            versionString.starts(with: "v"),
            let v = Int(versionString.dropFirst())
        {
            return v
        }
        return nil
    }

    public init(
        id: String,
        entryPoint: @escaping EntryPoint,
        displayName: String? = nil,
        description: String? = nil,
        category: EnvCategory? = nil,
        rewardThreshold: Double? = nil,
        nondeterministic: Bool = false,
        maxEpisodeSteps: Int? = nil,
        orderEnforce: Bool = true,
        disableEnvChecker: Bool = false,
        options: EnvOptions = [:],
        additionalWrappers: [WrapperSpec] = []
    ) {
        self.id = id
        self.entryPoint = entryPoint
        self.displayName = displayName
        self.description = description
        self.category = category
        self.rewardThreshold = rewardThreshold
        self.nondeterministic = nondeterministic
        self.maxEpisodeSteps = maxEpisodeSteps
        self.orderEnforce = orderEnforce
        self.disableEnvChecker = disableEnvChecker
        self.options = options
        self.additionalWrappers = additionalWrappers
    }
}

extension GymnazoRegistry {
    /// registers every environment bundled with Gymnazo.
    public func registerDefaultEnvironments() {
        registerToyText()
        registerClassicControl()
        registerBox2D()
    }

    private func registerIfNeeded(
        id: String,
        displayName: String,
        description: String,
        category: EnvCategory,
        entryPoint: @escaping @Sendable (EnvOptions) throws -> any Env,
        maxEpisodeSteps: Int? = nil,
        rewardThreshold: Double? = nil,
        nondeterministic: Bool = false
    ) {
        guard !isRegistered(id) else { return }
        register(
            id: id,
            displayName: displayName,
            description: description,
            category: category,
            entryPoint: entryPoint,
            maxEpisodeSteps: maxEpisodeSteps,
            rewardThreshold: rewardThreshold,
            nondeterministic: nondeterministic
        )
    }

    private func registerToyText() {
        registerFrozenLake()
        registerBlackjack()
        registerTaxi()
        registerCliffWalking()
    }

    private func registerFrozenLake() {
        registerIfNeeded(
            id: "FrozenLake",
            displayName: "Frozen Lake",
            description: "Navigate a frozen lake from start to goal without falling into holes.",
            category: .toyText,
            entryPoint: { options in
                try RegistrationSupport.createFrozenLake(options: options, defaultMap: "4x4")
            },
            maxEpisodeSteps: 100
        )

        registerIfNeeded(
            id: "FrozenLake8x8",
            displayName: "Frozen Lake 8x8",
            description: "Navigate a larger frozen lake from start to goal without falling into holes.",
            category: .toyText,
            entryPoint: { options in
                try RegistrationSupport.createFrozenLake(options: options, defaultMap: "8x8")
            },
            maxEpisodeSteps: 200
        )
    }

    private func registerBlackjack() {
        registerIfNeeded(
            id: "Blackjack",
            displayName: "Blackjack",
            description: "Play Blackjack against a dealer by learning when to hit or stand.",
            category: .toyText,
            entryPoint: { options in
                let renderMode = RegistrationSupport.renderMode(from: options)
                let natural = options["natural"] as? Bool ?? false
                let sab = options["sab"] as? Bool ?? false
                return Blackjack(renderMode: renderMode, natural: natural, sab: sab)
            }
        )
    }

    private func registerTaxi() {
        registerIfNeeded(
            id: "Taxi",
            displayName: "Taxi",
            description: "Navigate a taxi to pick up and drop off passengers at designated locations.",
            category: .toyText,
            entryPoint: { options in
                let renderMode = RegistrationSupport.renderMode(from: options)
                let isRainy = options["is_rainy"] as? Bool ?? false
                let ficklePassenger = options["fickle_passenger"] as? Bool ?? false
                return Taxi(
                    renderMode: renderMode,
                    isRainy: isRainy,
                    ficklePassenger: ficklePassenger
                )
            },
            maxEpisodeSteps: 200
        )
    }

    private func registerCliffWalking() {
        registerIfNeeded(
            id: "CliffWalking",
            displayName: "Cliff Walking",
            description: "Reach the end while avoiding falling off the cliff.",
            category: .toyText,
            entryPoint: { options in
                let renderMode = RegistrationSupport.renderMode(from: options)
                let isSlippery = options["is_slippery"] as? Bool ?? false
                return CliffWalking(renderMode: renderMode, isSlippery: isSlippery)
            },
            maxEpisodeSteps: 200
        )
    }

    private func registerClassicControl() {
        registerIfNeeded(
            id: "CartPole",
            displayName: "Cart Pole",
            description: "Balance a pole on a moving cart by applying left or right forces.",
            category: .classicControl,
            entryPoint: { options in
                let renderMode = RegistrationSupport.renderMode(from: options)
                return CartPole(renderMode: renderMode)
            },
            maxEpisodeSteps: 500
        )

        registerIfNeeded(
            id: "MountainCar",
            displayName: "Mountain Car",
            description: "Drive a car up a steep hill using momentum.",
            category: .classicControl,
            entryPoint: { options in
                let renderMode = RegistrationSupport.renderMode(from: options)
                let goalVelocity = RegistrationSupport.floatValue(
                    from: options["goal_velocity"],
                    default: 0.0
                )
                return MountainCar(renderMode: renderMode, goal_velocity: goalVelocity)
            },
            maxEpisodeSteps: 200
        )

        registerIfNeeded(
            id: "MountainCarContinuous",
            displayName: "Mountain Car Continuous",
            description: "Drive a car up a steep hill using continuous force.",
            category: .classicControl,
            entryPoint: { options in
                let renderMode = RegistrationSupport.renderMode(from: options)
                let goalVelocity = RegistrationSupport.floatValue(
                    from: options["goal_velocity"],
                    default: 0.0
                )
                return MountainCarContinuous(
                    renderMode: renderMode,
                    goal_velocity: goalVelocity
                )
            },
            maxEpisodeSteps: 999
        )

        registerIfNeeded(
            id: "Acrobot",
            displayName: "Acrobot",
            description: "Swing a two-link robotic arm to reach target height.",
            category: .classicControl,
            entryPoint: { options in
                let renderMode = RegistrationSupport.renderMode(from: options)
                let torqueNoiseMax = RegistrationSupport.floatValue(
                    from: options["torque_noise_max"],
                    default: 0.0
                )
                return Acrobot(renderMode: renderMode, torque_noise_max: torqueNoiseMax)
            },
            maxEpisodeSteps: 500,
            rewardThreshold: -100
        )

        registerIfNeeded(
            id: "Pendulum",
            displayName: "Pendulum",
            description: "Swing up and balance an inverted pendulum using torque.",
            category: .classicControl,
            entryPoint: { options in
                let renderMode = RegistrationSupport.renderMode(from: options)
                let g = RegistrationSupport.floatValue(from: options["g"], default: 10.0)
                return Pendulum(renderMode: renderMode, g: g)
            },
            maxEpisodeSteps: 200
        )
    }

    private func registerBox2D() {
        registerIfNeeded(
            id: "LunarLander",
            displayName: "Lunar Lander",
            description: "Safely land a rocket on the moon by controlling thrusters.",
            category: .box2D,
            entryPoint: { options in
                let settings = RegistrationSupport.lunarLanderSettings(from: options)
                return try LunarLander(
                    renderMode: settings.renderMode,
                    gravity: settings.gravity,
                    enableWind: settings.enableWind,
                    windPower: settings.windPower,
                    turbulencePower: settings.turbulencePower
                )
            },
            maxEpisodeSteps: 1000
        )

        registerIfNeeded(
            id: "LunarLanderContinuous",
            displayName: "Lunar Lander Continuous",
            description: "Safely land a rocket on the moon using continuous thrust control.",
            category: .box2D,
            entryPoint: { options in
                let settings = RegistrationSupport.lunarLanderSettings(from: options)
                return try LunarLanderContinuous(
                    renderMode: settings.renderMode,
                    gravity: settings.gravity,
                    enableWind: settings.enableWind,
                    windPower: settings.windPower,
                    turbulencePower: settings.turbulencePower
                )
            },
            maxEpisodeSteps: 1000
        )

        registerIfNeeded(
            id: "CarRacing",
            displayName: "Car Racing",
            description: "Race a car around a generated track using continuous controls.",
            category: .box2D,
            entryPoint: { options in
                let settings = RegistrationSupport.carRacingSettings(from: options)
                return CarRacing(
                    renderMode: settings.renderMode,
                    lapCompletePercent: settings.lapCompletePercent,
                    domainRandomize: settings.domainRandomize
                )
            },
            maxEpisodeSteps: 1000,
            rewardThreshold: 900
        )

        registerIfNeeded(
            id: "CarRacingDiscrete",
            displayName: "Car Racing Discrete",
            description: "Race a car around a generated track using discrete controls.",
            category: .box2D,
            entryPoint: { options in
                let settings = RegistrationSupport.carRacingSettings(from: options)
                return CarRacingDiscrete(
                    renderMode: settings.renderMode,
                    lapCompletePercent: settings.lapCompletePercent,
                    domainRandomize: settings.domainRandomize
                )
            },
            maxEpisodeSteps: 1000,
            rewardThreshold: 900
        )
    }
}

private struct RegistrationSupport {
    static func createFrozenLake(options: EnvOptions, defaultMap: String) throws -> FrozenLake {
        let renderMode = renderMode(from: options)
        let mapName = options["map_name"] as? String ?? defaultMap
        let isSlippery = options["is_slippery"] as? Bool ?? true
        let successRate = floatValue(from: options["success_rate"], default: 1.0 / 3.0)
        let desc = options["desc"] as? [String]

        return try FrozenLake(
            renderMode: renderMode,
            desc: desc,
            map_name: mapName,
            isSlippery: isSlippery,
            successRate: successRate
        )
    }

    static func renderMode(from options: EnvOptions) -> RenderMode? {
        if let mode = options["render_mode"] as? RenderMode {
            return mode
        }
        if let raw = options["render_mode"] as? String {
            return RenderMode(rawValue: raw)
        }
        return nil
    }

    static func lunarLanderSettings(from options: EnvOptions) -> (
        renderMode: RenderMode?,
        gravity: Float,
        enableWind: Bool,
        windPower: Float,
        turbulencePower: Float
    ) {
        let renderMode = renderMode(from: options)
        let gravity = floatValue(from: options["gravity"], default: -10.0)
        let enableWind = options["enable_wind"] as? Bool ?? false
        let windPower = floatValue(from: options["wind_power"], default: 15.0)
        let turbulencePower = floatValue(from: options["turbulence_power"], default: 1.5)
        return (renderMode, gravity, enableWind, windPower, turbulencePower)
    }

    static func carRacingSettings(from options: EnvOptions) -> (
        renderMode: RenderMode?,
        lapCompletePercent: Float,
        domainRandomize: Bool
    ) {
        let renderMode = renderMode(from: options)
        let lapCompletePercent = floatValue(
            from: options["lap_complete_percent"],
            default: 0.95
        )
        let domainRandomize = options["domain_randomize"] as? Bool ?? false
        return (renderMode, lapCompletePercent, domainRandomize)
    }

    static func floatValue(from value: (any Sendable)?, default defaultValue: Float) -> Float {
        switch value {
        case let float as Float:
            return float
        case let double as Double:
            return Float(double)
        case let int as Int:
            return Float(int)
        default:
            return defaultValue
        }
    }
}
