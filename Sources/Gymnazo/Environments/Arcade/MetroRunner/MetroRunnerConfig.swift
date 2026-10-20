import Foundation

/// Training subsets of the supplied Metro Run layouts.
public enum MetroRunnerCurriculum: String, Sendable, CaseIterable {
    case barriers, `static`, moving, full
}

/// Reward weights applied once per policy decision.
public struct MetroRunnerReward: Sendable, Equatable {
    public var distanceScale: Double
    public var coinWeight: Double
    public var deathPenalty: Double

    public init(distanceScale: Double = 0.01, coinWeight: Double = 0.05, deathPenalty: Double = 5) {
        self.distanceScale = distanceScale
        self.coinWeight = coinWeight
        self.deathPenalty = deathPenalty
    }
}

/// Settings for the native Metro Run approximation.
public struct MetroRunnerConfig: Sendable, Equatable {
    public var curriculum: MetroRunnerCurriculum
    public var frameSkip: Int
    public var maxEpisodeSeconds: Double
    public var reward: MetroRunnerReward
    public var section: String?

    public init(
        curriculum: MetroRunnerCurriculum = .full,
        frameSkip: Int = 4,
        maxEpisodeSeconds: Double = 300,
        reward: MetroRunnerReward = MetroRunnerReward(),
        section: String? = nil
    ) {
        self.curriculum = curriculum
        self.frameSkip = frameSkip
        self.maxEpisodeSeconds = maxEpisodeSeconds
        self.reward = reward
        self.section = section
    }

    func validate() throws {
        guard frameSkip > 0, maxEpisodeSeconds.isFinite, maxEpisodeSeconds > 0,
            (maxEpisodeSeconds * 60).rounded(.up) < Double(Int.max),
            reward.distanceScale.isFinite, reward.coinWeight.isFinite,
            reward.deathPenalty.isFinite
        else {
            throw GymnazoError.invalidConfiguration("MetroRunner requires positive frameSkip and finite positive duration and finite reward weights.")
        }
    }

    static func from(options: EnvOptions) throws -> Self {
        let supported: Set<String> = ["config", "render_mode"]
        guard Set(options.storage.keys).isSubset(of: supported) else {
            throw GymnazoError.invalidConfiguration("MetroRunner accepts config and render_mode creation options.")
        }
        if let value = options["config"] {
            guard let config = value as? Self else {
                throw GymnazoError.invalidConfiguration("MetroRunner config must be MetroRunnerConfig.")
            }
            return config
        }
        return Self()
    }
}

/// Discrete swipe-style commands. A command is applied only on the first physics tick.
public enum MetroRunnerAction: Int, Sendable, CaseIterable {
    case none, left, right, jump, down, board
}
