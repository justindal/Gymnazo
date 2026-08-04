/// Registration metadata and construction closure for a typed environment.
public struct ValueEnvSpec: Sendable {
    public typealias EntryPoint = @Sendable (EnvOptions) throws -> AnyValueEnvironment

    public let id: String
    public let entryPoint: EntryPoint
    public let displayName: String?
    public let description: String?
    public let category: EnvCategory?
    public let metadata: EnvMetadata
    public let maxEpisodeSteps: Int?
    public let rewardThreshold: Double?
    public let nondeterministic: Bool

    /// Creates a typed environment specification.
    ///
    /// - Parameters:
    ///   - id: Stable environment identifier.
    ///   - entryPoint: Closure that constructs the actor-owned environment.
    ///   - displayName: Optional human-readable name.
    ///   - description: Optional environment description.
    ///   - category: Optional environment category.
    ///   - metadata: Optional rendering metadata.
    ///   - maxEpisodeSteps: Optional default episode limit.
    ///   - rewardThreshold: Optional reward threshold.
    ///   - nondeterministic: Whether seeded execution may still vary.
    public init(
        id: String,
        entryPoint: @escaping EntryPoint,
        displayName: String? = nil,
        description: String? = nil,
        category: EnvCategory? = nil,
        metadata: EnvMetadata? = nil,
        maxEpisodeSteps: Int? = nil,
        rewardThreshold: Double? = nil,
        nondeterministic: Bool = false
    ) {
        self.id = id
        self.entryPoint = entryPoint
        self.displayName = displayName
        self.description = description
        self.category = category
        self.metadata = metadata ?? EnvMetadata.bundled(id: id)
        self.maxEpisodeSteps = maxEpisodeSteps
        self.rewardThreshold = rewardThreshold
        self.nondeterministic = nondeterministic
    }
}

/// Actor-isolated registry for typed value environments.
public actor ValueGymnazoRegistry {
    public static let shared = ValueGymnazoRegistry()

    private var specs: [String: ValueEnvSpec] = [:]
    private var initialized = false

    /// Creates an empty registry that installs bundled environments lazily.
    public init() {}

    /// Registers or replaces a typed environment specification.
    ///
    /// - Parameter spec: The specification to register.
    public func register(_ spec: ValueEnvSpec) {
        specs[spec.id] = spec
    }

    /// Returns registered ids in stable sorted order.
    public func ids() -> [String] {
        initializeIfNeeded()
        return specs.keys.sorted()
    }

    /// Returns all registered typed environment specifications.
    public func environmentSpecs() -> [String: ValueEnvSpec] {
        initializeIfNeeded()
        return specs
    }

    /// Returns whether an id is registered.
    ///
    /// - Parameter id: The environment identifier to check.
    /// - Returns: `true` when the registry contains the identifier.
    public func isRegistered(_ id: String) -> Bool {
        initializeIfNeeded()
        return specs[id] != nil
    }

    /// Constructs an actor-owned typed environment by id.
    ///
    /// - Parameters:
    ///   - id: Registered environment identifier.
    ///   - maxEpisodeSteps: Optional override for the episode limit.
    ///   - options: Constructor options passed to the entry point.
    /// - Returns: An actor-owned typed environment.
    /// - Throws: If the id is unregistered or construction fails.
    public func make(
        _ id: String,
        maxEpisodeSteps: Int? = nil,
        options: EnvOptions = [:]
    ) async throws -> AnyValueEnvironment {
        initializeIfNeeded()
        guard let spec = specs[id] else {
            throw GymnazoError.unregisteredEnvironment(id: id)
        }
        let environment = try spec.entryPoint(options)
        try await environment.applyTimeLimit(maxEpisodeSteps ?? spec.maxEpisodeSteps)
        return environment
    }

    private func initializeIfNeeded() {
        guard !initialized else { return }
        initialized = true

        let cartPole = ValueEnvSpec(
            id: "CartPole",
            entryPoint: Self.cartPoleEntryPoint(id: "CartPole"),
            displayName: "Cart Pole",
            description: "Balance a pole on a moving cart by applying left or right forces.",
            category: .classicControl,
            maxEpisodeSteps: 500,
            rewardThreshold: 475
        )
        let mountainCar = ValueEnvSpec(
            id: "MountainCar",
            entryPoint: Self.mountainCarEntryPoint(id: "MountainCar"),
            displayName: "Mountain Car",
            description: "Drive a car up a steep hill using momentum.",
            category: .classicControl,
            maxEpisodeSteps: 200,
            rewardThreshold: -110
        )
        let pendulum = ValueEnvSpec(
            id: "Pendulum",
            entryPoint: Self.pendulumEntryPoint(id: "Pendulum"),
            displayName: "Pendulum",
            description: "Swing an underactuated pendulum upright with continuous torque.",
            category: .classicControl,
            maxEpisodeSteps: 200
        )
        let acrobot = ValueEnvSpec(
            id: "Acrobot",
            entryPoint: Self.acrobotEntryPoint(id: "Acrobot"),
            displayName: "Acrobot",
            description: "Swing a two-link underactuated system above a target height.",
            category: .classicControl,
            maxEpisodeSteps: 500,
            rewardThreshold: -100
        )
        let mountainCarContinuous = ValueEnvSpec(
            id: "MountainCarContinuous",
            entryPoint: Self.mountainCarContinuousEntryPoint(
                id: "MountainCarContinuous"
            ),
            displayName: "Mountain Car Continuous",
            description: "Drive a car up a steep hill using continuous force.",
            category: .classicControl,
            maxEpisodeSteps: 999,
            rewardThreshold: 90
        )

        specs[cartPole.id] = cartPole
        specs[mountainCar.id] = mountainCar
        specs[pendulum.id] = pendulum
        specs[acrobot.id] = acrobot
        specs[mountainCarContinuous.id] = mountainCarContinuous
        installBundledEnvironmentSpecs()
    }

    private func installBundledEnvironmentSpecs() {
        install(
            ValueEnvSpec(
                id: "FrozenLake",
                entryPoint: Self.frozenLakeEntryPoint(id: "FrozenLake"),
                displayName: "Frozen Lake",
                description: "Navigate a frozen lake without falling into holes.",
                category: .toyText,
                maxEpisodeSteps: 100
            )
        )
        install(
            ValueEnvSpec(
                id: "FrozenLake8x8",
                entryPoint: Self.frozenLakeEntryPoint(
                    id: "FrozenLake8x8",
                    defaultMap: "8x8"
                ),
                displayName: "Frozen Lake 8x8",
                description: "Navigate a larger frozen lake without falling into holes.",
                category: .toyText,
                maxEpisodeSteps: 200
            )
        )
        install(
            ValueEnvSpec(
                id: "Blackjack",
                entryPoint: Self.blackjackEntryPoint(id: "Blackjack"),
                displayName: "Blackjack",
                description: "Learn when to hit or stand against a dealer.",
                category: .toyText
            )
        )
        install(
            ValueEnvSpec(
                id: "Taxi",
                entryPoint: Self.taxiEntryPoint(id: "Taxi"),
                displayName: "Taxi",
                description: "Pick up and deliver passengers in a grid world.",
                category: .toyText,
                maxEpisodeSteps: 200
            )
        )
        install(
            ValueEnvSpec(
                id: "CliffWalking",
                entryPoint: Self.cliffWalkingEntryPoint(id: "CliffWalking"),
                displayName: "Cliff Walking",
                description: "Reach the goal while avoiding the cliff.",
                category: .toyText,
                maxEpisodeSteps: 200
            )
        )
        install(
            ValueEnvSpec(
                id: "FlappyBird",
                entryPoint: Self.flappyBirdEntryPoint(id: "FlappyBird"),
                displayName: "Flappy Bird",
                description: "Fly through moving pipe gaps by choosing when to flap.",
                category: .arcade,
                maxEpisodeSteps: 10_000
            )
        )
        install(
            ValueEnvSpec(
                id: "Snake",
                entryPoint: Self.snakeEntryPoint(id: "Snake"),
                displayName: "Snake",
                description: "Grow a snake by collecting food without hitting walls or its body.",
                category: .arcade,
                maxEpisodeSteps: 1_000
            )
        )
        install(
            ValueEnvSpec(
                id: "2048",
                entryPoint: Self.twentyFortyEightEntryPoint(id: "2048"),
                displayName: "2048",
                description: "Combine matching tiles to reach the highest possible value.",
                category: .arcade,
                maxEpisodeSteps: 10_000
            )
        )
        install(
            ValueEnvSpec(
                id: "LunarLander",
                entryPoint: Self.lunarLanderEntryPoint(
                    id: "LunarLander",
                    continuous: false
                ),
                displayName: "Lunar Lander",
                description: "Land a rocket using discrete thruster controls.",
                category: .box2D,
                maxEpisodeSteps: 1000
            )
        )
        install(
            ValueEnvSpec(
                id: "LunarLanderContinuous",
                entryPoint: Self.lunarLanderEntryPoint(
                    id: "LunarLanderContinuous",
                    continuous: true
                ),
                displayName: "Lunar Lander Continuous",
                description: "Land a rocket using continuous thruster controls.",
                category: .box2D,
                maxEpisodeSteps: 1000
            )
        )
        install(
            ValueEnvSpec(
                id: "CarRacing",
                entryPoint: Self.carRacingEntryPoint(
                    id: "CarRacing",
                    continuous: true
                ),
                displayName: "Car Racing",
                description: "Race around a generated track with continuous controls.",
                category: .box2D,
                maxEpisodeSteps: 1000,
                rewardThreshold: 900
            )
        )
        install(
            ValueEnvSpec(
                id: "CarRacingDiscrete",
                entryPoint: Self.carRacingEntryPoint(
                    id: "CarRacingDiscrete",
                    continuous: false
                ),
                displayName: "Car Racing Discrete",
                description: "Race around a generated track with discrete controls.",
                category: .box2D,
                maxEpisodeSteps: 1000,
                rewardThreshold: 900
            )
        )
    }

    private func install(_ spec: ValueEnvSpec) {
        specs[spec.id] = spec
    }

    private static func cartPoleEntryPoint(id: String) -> ValueEnvSpec.EntryPoint {
        { options in
            guard options.isEmpty else {
                throw GymnazoError.invalidConfiguration(
                    "CartPole does not support creation options.")
            }
            return AnyValueEnvironment(id: id, try CartPoleValueEnvironment())
        }
    }

    private static func flappyBirdEntryPoint(id: String) -> ValueEnvSpec.EntryPoint {
        { options in
            guard options.isEmpty else {
                throw GymnazoError.invalidConfiguration(
                    "FlappyBird does not support creation options."
                )
            }
            return AnyValueEnvironment(id: id, try FlappyBirdValueEnvironment())
        }
    }

    private static func snakeEntryPoint(id: String) -> ValueEnvSpec.EntryPoint {
        { options in
            guard options.isEmpty else {
                throw GymnazoError.invalidConfiguration(
                    "Snake does not support creation options."
                )
            }
            return AnyValueEnvironment(id: id, try SnakeValueEnvironment())
        }
    }

    private static func twentyFortyEightEntryPoint(id: String) -> ValueEnvSpec.EntryPoint {
        { options in
            guard options.isEmpty else {
                throw GymnazoError.invalidConfiguration(
                    "2048 does not support creation options."
                )
            }
            return AnyValueEnvironment(id: id, try TwentyFortyEightValueEnvironment())
        }
    }

    private static func mountainCarEntryPoint(id: String) -> ValueEnvSpec.EntryPoint {
        { options in
            let supported = Set(["goal_velocity"])
            let unsupported = Set(options.storage.keys).subtracting(supported)
            guard unsupported.isEmpty else {
                throw GymnazoError.invalidConfiguration(
                    "MountainCar does not support creation options: \(unsupported.sorted().joined(separator: ", "))."
                )
            }
            let goalVelocity = try floatValue(
                options["goal_velocity"],
                key: "goal_velocity",
                default: 0
            )
            return AnyValueEnvironment(
                id: id,
                try MountainCarValueEnvironment(goalVelocity: goalVelocity)
            )
        }
    }

    private static func pendulumEntryPoint(id: String) -> ValueEnvSpec.EntryPoint {
        { options in
            let supported = Set(["g"])
            let unsupported = Set(options.storage.keys).subtracting(supported)
            guard unsupported.isEmpty else {
                throw GymnazoError.invalidConfiguration(
                    "Pendulum does not support creation options: \(unsupported.sorted().joined(separator: ", "))."
                )
            }
            let gravity = try floatValue(options["g"], key: "g", default: 10)
            return AnyValueEnvironment(
                id: id,
                try PendulumValueEnvironment(g: gravity)
            )
        }
    }

    private static func acrobotEntryPoint(id: String) -> ValueEnvSpec.EntryPoint {
        { options in
            let supported = Set(["torque_noise_max", "book_or_nips"])
            let unsupported = Set(options.storage.keys).subtracting(supported)
            guard unsupported.isEmpty else {
                throw GymnazoError.invalidConfiguration(
                    "Acrobot does not support creation options: \(unsupported.sorted().joined(separator: ", "))."
                )
            }
            let noise = try floatValue(
                options["torque_noise_max"],
                key: "torque_noise_max",
                default: 0
            )
            let mode = options["book_or_nips"] as? String ?? "book"
            guard mode == "book" || mode == "nips" else {
                throw GymnazoError.invalidConfiguration(
                    "book_or_nips must be book or nips."
                )
            }
            return AnyValueEnvironment(
                id: id,
                try AcrobotValueEnvironment(
                    torqueNoiseMax: noise,
                    useNipsDynamics: mode == "nips"
                )
            )
        }
    }

    private static func mountainCarContinuousEntryPoint(
        id: String
    ) -> ValueEnvSpec.EntryPoint {
        { options in
            let supported = Set(["goal_velocity"])
            let unsupported = Set(options.storage.keys).subtracting(supported)
            guard unsupported.isEmpty else {
                throw GymnazoError.invalidConfiguration(
                    "MountainCarContinuous does not support creation options: \(unsupported.sorted().joined(separator: ", "))."
                )
            }
            let goalVelocity = try floatValue(
                options["goal_velocity"],
                key: "goal_velocity",
                default: 0
            )
            return AnyValueEnvironment(
                id: id,
                try MountainCarContinuousValueEnvironment(goalVelocity: goalVelocity)
            )
        }
    }

    private static func frozenLakeEntryPoint(
        id: String,
        defaultMap: String = "4x4"
    ) -> ValueEnvSpec.EntryPoint {
        { options in
            try validateOptions(
                options,
                supported: ["desc", "map_name", "is_slippery", "success_rate"]
            )
            let description = options["desc"] as? [String]
            let mapName = options["map_name"] as? String ?? defaultMap
            let slippery = try boolValue(
                options["is_slippery"],
                key: "is_slippery",
                default: true
            )
            let successRate = try floatValue(
                options["success_rate"],
                key: "success_rate",
                default: 1 / 3
            )
            guard (0...1).contains(successRate) else {
                throw GymnazoError.invalidConfiguration(
                    "success_rate must be between zero and one."
                )
            }
            return AnyValueEnvironment(
                id: id,
                try FrozenLakeValueEnvironment(
                    desc: description,
                    mapName: mapName,
                    isSlippery: slippery,
                    successRate: successRate
                )
            )
        }
    }

    private static func blackjackEntryPoint(id: String) -> ValueEnvSpec.EntryPoint {
        { options in
            try validateOptions(options, supported: ["natural", "sab"])
            let natural = try boolValue(options["natural"], key: "natural", default: false)
            let sab = try boolValue(options["sab"], key: "sab", default: false)
            return AnyValueEnvironment(
                id: id,
                try BlackjackValueEnvironment(natural: natural, sab: sab)
            )
        }
    }

    private static func taxiEntryPoint(id: String) -> ValueEnvSpec.EntryPoint {
        { options in
            try validateOptions(options, supported: ["is_rainy", "fickle_passenger"])
            let rainy = try boolValue(options["is_rainy"], key: "is_rainy", default: false)
            let fickle = try boolValue(
                options["fickle_passenger"],
                key: "fickle_passenger",
                default: false
            )
            return AnyValueEnvironment(
                id: id,
                try TaxiValueEnvironment(isRainy: rainy, ficklePassenger: fickle)
            )
        }
    }

    private static func cliffWalkingEntryPoint(id: String) -> ValueEnvSpec.EntryPoint {
        { options in
            try validateOptions(options, supported: ["is_slippery"])
            let slippery = try boolValue(
                options["is_slippery"],
                key: "is_slippery",
                default: false
            )
            return AnyValueEnvironment(
                id: id,
                try CliffWalkingValueEnvironment(isSlippery: slippery)
            )
        }
    }

    private static func lunarLanderEntryPoint(
        id: String,
        continuous: Bool
    ) -> ValueEnvSpec.EntryPoint {
        { options in
            try validateOptions(
                options,
                supported: ["gravity", "enable_wind", "wind_power", "turbulence_power"]
            )
            let gravity = try floatValue(options["gravity"], key: "gravity", default: -10)
            let enableWind = try boolValue(
                options["enable_wind"],
                key: "enable_wind",
                default: false
            )
            let windPower = try floatValue(
                options["wind_power"],
                key: "wind_power",
                default: 15
            )
            let turbulencePower = try floatValue(
                options["turbulence_power"],
                key: "turbulence_power",
                default: 1.5
            )
            if continuous {
                let base = try LunarLanderContinuous(
                    gravity: gravity,
                    enableWind: enableWind,
                    windPower: windPower,
                    turbulencePower: turbulencePower
                )
                let storage = LunarLanderContinuousValueStorage(base)
                return AnyValueEnvironment(
                    id: id,
                    storage: storage,
                    observationSpace: AnyValueSpace(try lunarLanderObservationSpace()),
                    actionSpace: AnyValueSpace(
                        try BoxValueSpace(
                            low: try TensorValue([-1, -1], shape: [2]),
                            high: try TensorValue([1, 1], shape: [2])
                        )
                    )
                )
            }
            let base = try LunarLander(
                gravity: gravity,
                enableWind: enableWind,
                windPower: windPower,
                turbulencePower: turbulencePower
            )
            let storage = LunarLanderValueStorage(base)
            return AnyValueEnvironment(
                id: id,
                storage: storage,
                observationSpace: AnyValueSpace(try lunarLanderObservationSpace()),
                actionSpace: AnyValueSpace(try DiscreteValueSpace(n: 4))
            )
        }
    }

    private static func carRacingEntryPoint(
        id: String,
        continuous: Bool
    ) -> ValueEnvSpec.EntryPoint {
        { options in
            try validateOptions(
                options,
                supported: ["lap_complete_percent", "domain_randomize"]
            )
            let lapComplete = try floatValue(
                options["lap_complete_percent"],
                key: "lap_complete_percent",
                default: 0.95
            )
            guard (0...1).contains(lapComplete) else {
                throw GymnazoError.invalidConfiguration(
                    "lap_complete_percent must be between zero and one."
                )
            }
            let randomize = try boolValue(
                options["domain_randomize"],
                key: "domain_randomize",
                default: false
            )
            if continuous {
                let base = CarRacing(
                    lapCompletePercent: lapComplete,
                    domainRandomize: randomize
                )
                let storage = CarRacingValueStorage(base)
                return AnyValueEnvironment(
                    id: id,
                    storage: storage,
                    observationSpace: AnyValueSpace(
                        try BoxValueSpace(low: 0, high: 255, shape: [96, 96, 3])
                    ),
                    actionSpace: AnyValueSpace(
                        try BoxValueSpace(
                            low: try TensorValue([-1, 0, 0], shape: [3]),
                            high: try TensorValue([1, 1, 1], shape: [3])
                        )
                    )
                )
            }
            let base = CarRacingDiscrete(
                lapCompletePercent: lapComplete,
                domainRandomize: randomize
            )
            let storage = CarRacingDiscreteValueStorage(base)
            return AnyValueEnvironment(
                id: id,
                storage: storage,
                observationSpace: AnyValueSpace(
                    try BoxValueSpace(low: 0, high: 255, shape: [96, 96, 3])
                ),
                actionSpace: AnyValueSpace(try DiscreteValueSpace(n: 5))
            )
        }
    }

    private static func lunarLanderObservationSpace() throws -> BoxValueSpace {
        try BoxValueSpace(
            low: try TensorValue(
                [-2.5, -2.5, -10, -10, -2 * .pi, -10, 0, 0],
                shape: [8]
            ),
            high: try TensorValue(
                [2.5, 2.5, 10, 10, 2 * .pi, 10, 1, 1],
                shape: [8]
            )
        )
    }

    private static func validateOptions(
        _ options: EnvOptions,
        supported: Set<String>
    ) throws {
        let unsupported = Set(options.storage.keys).subtracting(supported)
        guard unsupported.isEmpty else {
            throw GymnazoError.invalidConfiguration(
                "Unsupported creation options: \(unsupported.sorted().joined(separator: ", "))."
            )
        }
    }

    private static func boolValue(
        _ value: (any Sendable)?,
        key: String,
        default defaultValue: Bool
    ) throws -> Bool {
        guard let value else { return defaultValue }
        guard let result = value as? Bool else {
            throw GymnazoError.invalidConfiguration("\(key) must be a Boolean.")
        }
        return result
    }

    private static func floatValue(
        _ value: (any Sendable)?,
        key: String,
        default defaultValue: Float
    ) throws -> Float {
        let result: Float
        switch value {
        case nil:
            result = defaultValue
        case let float as Float:
            result = float
        case let double as Double:
            result = Float(double)
        case let int as Int:
            result = Float(int)
        default:
            throw GymnazoError.invalidConfiguration("\(key) must be numeric.")
        }
        guard result.isFinite else {
            throw GymnazoError.invalidConfiguration("\(key) must be finite.")
        }
        return result
    }
}

extension Gymnazo {
    /// Returns all registered typed environment specifications.
    ///
    /// - Returns: Specifications keyed by stable environment identifier.
    public static func valueRegistry() async -> [String: ValueEnvSpec] {
        await ValueGymnazoRegistry.shared.environmentSpecs()
    }

    /// Returns registered typed environment ids in stable sorted order.
    ///
    /// - Returns: Registered environment identifiers.
    public static func valueIDs() async -> [String] {
        await ValueGymnazoRegistry.shared.ids()
    }

    /// Returns whether a typed environment id is registered.
    ///
    /// - Parameter id: The environment identifier to check.
    /// - Returns: True when the identifier is registered.
    public static func isValueRegistered(_ id: String) async -> Bool {
        await ValueGymnazoRegistry.shared.isRegistered(id)
    }

    /// Registers or replaces a typed environment specification.
    ///
    /// - Parameter spec: The specification to register.
    public static func registerValue(_ spec: ValueEnvSpec) async {
        await ValueGymnazoRegistry.shared.register(spec)
    }

    /// Creates one actor-owned typed environment by id.
    ///
    /// - Parameters:
    ///   - id: Registered environment identifier.
    ///   - maxEpisodeSteps: Optional override for the episode limit.
    ///   - options: Constructor options passed to the entry point.
    /// - Returns: An actor-owned typed environment.
    /// - Throws: If the id is unregistered or construction fails.
    public static func makeValue(
        _ id: String,
        maxEpisodeSteps: Int? = nil,
        options: EnvOptions = [:]
    ) async throws -> AnyValueEnvironment {
        try await ValueGymnazoRegistry.shared.make(
            id,
            maxEpisodeSteps: maxEpisodeSteps,
            options: options
        )
    }

    /// Creates a typed vector with one actor-owned environment per slot.
    ///
    /// - Parameters:
    ///   - id: Registered environment identifier.
    ///   - numEnvs: Number of environment slots to create.
    ///   - mode: Sequential or actor-owned parallel collection mode.
    ///   - autoresetMode: How completed slots are reset.
    ///   - maxEpisodeSteps: Optional episode limit applied to each slot.
    ///   - options: Constructor options passed to each entry point.
    /// - Returns: An actor-owned typed vector environment.
    /// - Throws: If `numEnvs` is invalid, the id is unregistered, or construction fails.
    public static func makeValueVec(
        _ id: String,
        numEnvs: Int,
        mode: ValueVectorMode = .sync,
        autoresetMode: AutoresetMode = .nextStep,
        maxEpisodeSteps: Int? = nil,
        options: EnvOptions = [:]
    ) async throws -> AnyValueVectorEnvironment {
        guard numEnvs > 0 else {
            throw GymnazoError.invalidNumEnvs(numEnvs)
        }
        var environments: [AnyValueEnvironment] = []
        environments.reserveCapacity(numEnvs)
        do {
            for _ in 0..<numEnvs {
                environments.append(
                    try await makeValue(
                        id,
                        maxEpisodeSteps: maxEpisodeSteps,
                        options: options
                    )
                )
            }
            return try AnyValueVectorEnvironment(
                environments: environments,
                mode: mode,
                autoresetMode: autoresetMode
            )
        } catch {
            for environment in environments {
                await environment.close()
            }
            throw error
        }
    }
}
