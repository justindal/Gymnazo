//
// Registration.swift
//

public typealias EnvCreator = ([String: Any]) -> any Env

public struct WrapperSpec {
    public let id: String
    /// entry point that, given an inner environment and kwargs, returns a wrapped environment.
    public let entryPoint: (any Env, [String: Any]) -> any Env
    public var kwargs: [String: Any] = [:]
}

public struct EnvSpec {
    public let id: String
    
    public enum EntryPoint {
        case creator(EnvCreator)
        case string(String)
    }

    public var entry_point: EntryPoint?

    // environment attributes
    public var rewardThreshold: Double? = nil
    public var nondeterministic: Bool = false

    // wrappers
    public var maxEpisodeSteps: Int? = nil
    public var order_enforce: Bool = true
    public var disable_env_checker: Bool = false

    // additional kwargs
    public var kwargs: [String: Any] = [:]

    // applied wrappers
    public var additional_wrappers: [WrapperSpec] = []

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
           let v = Int(versionString.dropFirst()) {
            return v
        }
        return nil
    }

    public init(
        id: String,
        entry_point: EntryPoint? = nil,
        rewardThreshold: Double? = nil,
        nondeterministic: Bool = false,
        maxEpisodeSteps: Int? = nil,
        order_enforce: Bool = true,
        disable_env_checker: Bool = false,
        kwargs: [String : Any] = [:],
        additional_wrappers: [WrapperSpec] = []
    ) {
        self.id = id
        self.entry_point = entry_point
        self.rewardThreshold = rewardThreshold
        self.nondeterministic = nondeterministic
        self.maxEpisodeSteps = maxEpisodeSteps
        self.order_enforce = order_enforce
        self.disable_env_checker = disable_env_checker
        self.kwargs = kwargs
        self.additional_wrappers = additional_wrappers
    }
}

/// register all built-in Gymnazo environments in one place.
@MainActor
public final class GymnazoRegistrations {
    public static let shared: GymnazoRegistrations = GymnazoRegistrations()

    public init(registerDefaults: Bool = true) {
        if registerDefaults {
            registerDefaultEnvironments()
        }
    }

    /// registers every environment bundled with Gymnazo.
    public func registerDefaultEnvironments() {
        registerToyText()
        registerClassicControl()
        registerBox2D()
    }
    
    private func registerToyText() {
        registerFrozenLake()
        registerBlackjack()
        registerTaxi()
        registerCliffWalking()
    }

    private func registerFrozenLake() {
        if registry["FrozenLake"] == nil {
            register(id: "FrozenLake", entryPoint: { kwargs in
                self.createFrozenLake(kwargs: kwargs, defaultMap: "4x4")
            }, maxEpisodeSteps: 100)
        }

        if registry["FrozenLake8x8"] == nil {
            register(id: "FrozenLake8x8", entryPoint: { kwargs in
                self.createFrozenLake(kwargs: kwargs, defaultMap: "8x8")
            }, maxEpisodeSteps: 200)
        }
    }
    
    private func registerBlackjack() {
        if registry["Blackjack"] == nil {
            register(id: "Blackjack", entryPoint: { kwargs in
                let renderMode = kwargs["render_mode"] as? String
                let natural = kwargs["natural"] as? Bool ?? false
                let sab = kwargs["sab"] as? Bool ?? false
                return Blackjack(render_mode: renderMode, natural: natural, sab: sab)
            })
        }
    }
    
    private func registerTaxi() {
        if registry["Taxi"] == nil {
            register(id: "Taxi", entryPoint: { kwargs in
                let renderMode = kwargs["render_mode"] as? String
                let isRainy = kwargs["is_rainy"] as? Bool ?? false
                let ficklePassenger = kwargs["fickle_passenger"] as? Bool ?? false
                return Taxi(render_mode: renderMode, isRainy: isRainy, ficklePassenger: ficklePassenger)
            }, maxEpisodeSteps: 200)
        }
    }
    
    private func registerCliffWalking() {
        if registry["CliffWalking"] == nil {
            register(id: "CliffWalking", entryPoint: { kwargs in
                let renderMode = kwargs["render_mode"] as? String
                let isSlippery = kwargs["is_slippery"] as? Bool ?? false
                return CliffWalking(render_mode: renderMode, isSlippery: isSlippery)
            }, maxEpisodeSteps: 200)
        }
    }
    
    private func registerClassicControl() {
        if registry["CartPole"] == nil {
            register(id: "CartPole", entryPoint: { kwargs in
                let renderMode = kwargs["render_mode"] as? String
                return CartPole(render_mode: renderMode)
            }, maxEpisodeSteps: 500)
        }
        
        if registry["MountainCar"] == nil {
            register(id: "MountainCar", entryPoint: { kwargs in
                let renderMode = kwargs["render_mode"] as? String
                let goalVelocity = GymnazoRegistrations.floatValue(
                    from: kwargs["goal_velocity"],
                    default: 0.0
                )
                return MountainCar(render_mode: renderMode, goal_velocity: goalVelocity)
            }, maxEpisodeSteps: 200)
        }
        
        if registry["MountainCarContinuous"] == nil {
            register(id: "MountainCarContinuous", entryPoint: { kwargs in
                let renderMode = kwargs["render_mode"] as? String
                let goalVelocity = GymnazoRegistrations.floatValue(
                    from: kwargs["goal_velocity"],
                    default: 0.0
                )
                return MountainCarContinuous(render_mode: renderMode, goal_velocity: goalVelocity)
            }, maxEpisodeSteps: 999)
        }
        
        if registry["Acrobot"] == nil {
            register(id: "Acrobot", entryPoint: { kwargs in
                let renderMode = kwargs["render_mode"] as? String
                return Acrobot(render_mode: renderMode)
            }, maxEpisodeSteps: 500, rewardThreshold: -100)
        }
        
        if registry["Pendulum"] == nil {
            register(id: "Pendulum", entryPoint: { kwargs in
                let renderMode = kwargs["render_mode"] as? String
                let g = GymnazoRegistrations.floatValue(from: kwargs["g"], default: 10.0)
                return Pendulum(render_mode: renderMode, g: g)
            }, maxEpisodeSteps: 200)
        }
    }
    
    private func registerBox2D() {
        if registry["LunarLander"] == nil {
            register(id: "LunarLander", entryPoint: { kwargs in
                let renderMode = kwargs["render_mode"] as? String
                let gravity = GymnazoRegistrations.floatValue(from: kwargs["gravity"], default: -10.0)
                let enableWind = kwargs["enable_wind"] as? Bool ?? false
                let windPower = GymnazoRegistrations.floatValue(from: kwargs["wind_power"], default: 15.0)
                let turbulencePower = GymnazoRegistrations.floatValue(from: kwargs["turbulence_power"], default: 1.5)
                
                return LunarLander(
                    render_mode: renderMode,
                    gravity: gravity,
                    enableWind: enableWind,
                    windPower: windPower,
                    turbulencePower: turbulencePower
                )
            }, maxEpisodeSteps: 1000)
        }
        
        if registry["LunarLanderContinuous"] == nil {
            register(id: "LunarLanderContinuous", entryPoint: { kwargs in
                let renderMode = kwargs["render_mode"] as? String
                let gravity = GymnazoRegistrations.floatValue(from: kwargs["gravity"], default: -10.0)
                let enableWind = kwargs["enable_wind"] as? Bool ?? false
                let windPower = GymnazoRegistrations.floatValue(from: kwargs["wind_power"], default: 15.0)
                let turbulencePower = GymnazoRegistrations.floatValue(from: kwargs["turbulence_power"], default: 1.5)
                
                return LunarLanderContinuous(
                    render_mode: renderMode,
                    gravity: gravity,
                    enableWind: enableWind,
                    windPower: windPower,
                    turbulencePower: turbulencePower
                )
            }, maxEpisodeSteps: 1000)
        }
    }
    
    private func createFrozenLake(kwargs: [String: Any], defaultMap: String) -> FrozenLake {
        let renderMode = kwargs["render_mode"] as? String
        let mapName = kwargs["map_name"] as? String ?? defaultMap
        let isSlippery = kwargs["is_slippery"] as? Bool ?? true
        let successRate = GymnazoRegistrations.floatValue(
            from: kwargs["success_rate"],
            default: Float(1.0 / 3.0)
        )

        // allow either a user-provided desc or a random map
        var desc = kwargs["desc"] as? [String]
        let useRandomMap = kwargs["generate_random_map"] as? Bool ?? false
        if desc == nil && useRandomMap {
            let size = (kwargs["size"] as? Int) ?? (mapName == "8x8" ? 8 : 4)
            let p = GymnazoRegistrations.floatValue(from: kwargs["p"], default: 0.8)
            desc = FrozenLake.generateRandomMap(size: size, p: p)
        }

        return FrozenLake(
            render_mode: renderMode,
            desc: desc,
            map_name: mapName,
            isSlippery: isSlippery,
            successRate: successRate
        )
    }

    private static func floatValue(from value: Any?, default defaultValue: Float) -> Float {
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