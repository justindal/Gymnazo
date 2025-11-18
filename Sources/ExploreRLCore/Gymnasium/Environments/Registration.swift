//
// Registration.swift
//

public typealias EnvCreator = ([String: Any]) -> any Environment

public struct WrapperSpec {
    public let id: String
    /// entry point that, given an inner environment and kwargs, returns a wrapped environment.
    public let entryPoint: (any Environment, [String: Any]) -> any Environment
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

/// register all built-in Gymnasium environments in one place.
@MainActor
public final class GymnasiumRegistrations {
    public static let shared: GymnasiumRegistrations = GymnasiumRegistrations()

    public init(registerDefaults: Bool = true) {
        if registerDefaults {
            registerDefaultEnvironments()
        }
    }

    /// registers every environment bundled with ExploreRLCore.
    public func registerDefaultEnvironments() {
        registerFrozenLake()
    }

    private func registerFrozenLake() {
        let envId: String = "FrozenLake-v1"
        guard Gymnasium.registry[envId] == nil else { return }

        Gymnasium.register(id: envId, entryPoint: { kwargs in
            let renderMode = kwargs["render_mode"] as? String
            let desc = kwargs["desc"] as? [String]
            let mapName = kwargs["map_name"] as? String ?? "4x4"
            let isSlippery = kwargs["is_slippery"] as? Bool ?? true
            let successRate = GymnasiumRegistrations.floatValue(
                from: kwargs["success_rate"],
                default: Float(1.0 / 3.0)
            )

            return FrozenLake(
                render_mode: renderMode,
                desc: desc,
                map_name: mapName,
                isSlippery: isSlippery,
                successRate: successRate
            )
        }, maxEpisodeSteps: 100)
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