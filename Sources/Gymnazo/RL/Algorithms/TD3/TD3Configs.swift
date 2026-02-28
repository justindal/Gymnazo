/// Network configuration for the ``TD3`` actor network.
public struct TD3ActorConfig: Sendable, Codable, Equatable {
    /// MLP architecture for the actor. Defaults to `[400, 300]`.
    public var netArch: NetArch
    /// Feature extractor applied before the MLP.
    public var featuresExtractor: FeaturesExtractorConfig
    /// Activation function used between MLP layers.
    public var activation: ActivationConfig
    /// Whether to normalize uint8 observations to `[0, 1]`.
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

/// Action-exploration noise applied at collection time for ``TD3``.
///
/// - `normal`: Independent Gaussian noise with the given standard deviation.
/// - `ornsteinUhlenbeck`: Temporally correlated noise suitable for environments with inertia.
public enum TD3ActionNoiseConfig: Sendable, Codable, Equatable {
    case normal(std: Float = 0.1)
    case ornsteinUhlenbeck(
        std: Float = 0.2,
        theta: Float = 0.15,
        dt: Float = 0.01,
        initialNoise: Float = 0.0
    )
}

/// TD3-specific algorithm hyper-parameters.
public struct TD3AlgorithmConfig: Sendable, Codable, Equatable {
    /// Number of critic updates between each actor update (default: 2).
    public let policyDelay: Int
    /// Std of the Gaussian noise added to target-policy actions during critic updates.
    public let targetPolicyNoise: Float
    /// Clip bound applied to the target-policy noise.
    public let targetNoiseClip: Float
    /// Optional exploration noise applied during environment interaction. `nil` disables noise.
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

/// Combined actor and critic network configuration for ``TD3``.
public struct TD3PolicyConfig: Sendable, Codable, Equatable {
    /// Optional MLP hidden-layer sizes shared by actor and critic.
    /// If `nil`, defaults to `[400, 300]` for flat observations or `[256, 256]` for CNN-based extractors.
    public var netArch: NetArch?
    /// Feature extractor applied before the actor and critic MLPs.
    public var featuresExtractor: FeaturesExtractorConfig
    /// Activation function used between MLP layers.
    public var activation: ActivationConfig
    /// Whether to normalize uint8 observations to `[0, 1]`.
    public var normalizeImages: Bool
    /// Number of critic Q-networks in the ensemble (default: 2).
    public var nCritics: Int
    /// Whether actor and critic share a feature extractor.
    public var shareFeaturesExtractor: Bool
    /// Optimizer configuration for the actor.
    public var actorOptimizer: OptimizerConfig
    /// Optimizer configuration for the critic.
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
