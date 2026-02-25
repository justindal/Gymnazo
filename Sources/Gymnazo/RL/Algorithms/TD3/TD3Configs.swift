//
//  TD3Configs.swift
//  Gymnazo
//
//  Created by Justin Daludado on 2026-02-23.
//

public struct TD3ActorConfig: Sendable, Codable, Equatable {
    public var netArch: NetArch
    public var featuresExtractor: FeaturesExtractorConfig
    public var activation: ActivationConfig
    public var normalizeImages: Bool

    public init(
        netArch: NetArch = .shared([400, 300]),
        featuresExtractor: FeaturesExtractorConfig = .auto,
        activation: ActivationConfig = .relu,
        normalizeImages: Bool = true
    ) {
        self.netArch = netArch
        self.featuresExtractor = featuresExtractor
        self.activation = activation
        self.normalizeImages = normalizeImages
    }
}

public enum TD3ActionNoiseConfig: Sendable, Codable, Equatable {
    case normal(std: Float = 0.1)
    case ornsteinUhlenbeck(
        std: Float = 0.2,
        theta: Float = 0.15,
        dt: Float = 0.01,
        initialNoise: Float = 0.0
    )
}

public struct TD3AlgorithmConfig: Sendable, Codable, Equatable {
    public let policyDelay: Int
    public let targetPolicyNoise: Float
    public let targetNoiseClip: Float
    public let actionNoise: TD3ActionNoiseConfig?

    public init(
        policyDelay: Int = 2,
        targetPolicyNoise: Float = 0.2,
        targetNoiseClip: Float = 0.5,
        actionNoise: TD3ActionNoiseConfig? = nil,
        actionNoiseStd: Float? = nil
    ) {
        self.policyDelay = max(1, policyDelay)
        self.targetPolicyNoise = max(0.0, targetPolicyNoise)
        self.targetNoiseClip = max(0.0, targetNoiseClip)
        if let actionNoise {
            self.actionNoise = actionNoise.sanitized()
        } else if let actionNoiseStd {
            let sanitizedStd = max(0.0, actionNoiseStd)
            self.actionNoise =
                sanitizedStd > 0
                ? .normal(std: sanitizedStd)
                : nil
        } else {
            self.actionNoise = nil
        }
    }
}

extension TD3ActionNoiseConfig {
    fileprivate func sanitized() -> TD3ActionNoiseConfig {
        switch self {
        case .normal(let std):
            return .normal(std: max(0.0, std))
        case .ornsteinUhlenbeck(let std, let theta, let dt, let initialNoise):
            return .ornsteinUhlenbeck(
                std: max(0.0, std),
                theta: max(0.0, theta),
                dt: max(1e-9, dt),
                initialNoise: initialNoise
            )
        }
    }
}

public struct TD3PolicyConfig: Sendable, Codable, Equatable {
    public var netArch: NetArch?
    public var featuresExtractor: FeaturesExtractorConfig
    public var activation: ActivationConfig
    public var normalizeImages: Bool
    public var nCritics: Int
    public var shareFeaturesExtractor: Bool
    public var actorOptimizer: OptimizerConfig
    public var criticOptimizer: OptimizerConfig

    public init(
        netArch: NetArch? = nil,
        featuresExtractor: FeaturesExtractorConfig = .auto,
        activation: ActivationConfig = .relu,
        normalizeImages: Bool = true,
        nCritics: Int = 2,
        shareFeaturesExtractor: Bool = false,
        actorOptimizer: OptimizerConfig = .adam(),
        criticOptimizer: OptimizerConfig = .adam()
    ) {
        self.netArch = netArch
        self.featuresExtractor = featuresExtractor
        self.activation = activation
        self.normalizeImages = normalizeImages
        self.nCritics = max(1, nCritics)
        self.shareFeaturesExtractor = shareFeaturesExtractor
        self.actorOptimizer = actorOptimizer
        self.criticOptimizer = criticOptimizer
    }
}
