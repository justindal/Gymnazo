import MLXOptimizers

/// Specifies which feature extractor to build for a policy network.
///
/// - `auto`: Selects `NatureCNN` for 3-D image observations, `FlattenExtractor` otherwise, and `CombinedExtractor` for `Dict` spaces.
/// - `flatten`: Always uses a `FlattenExtractor`.
/// - `natureCNN`: Uses a `NatureCNN` with the specified output dimension (image observations only).
/// - `combined`: Uses a `CombinedExtractor` for `Dict` observation spaces.
public enum FeaturesExtractorConfig: Sendable, Codable, Equatable {
    case auto
    case flatten
    case natureCNN(featuresDim: Int = 512)
    case combined(featuresDim: Int = 256, cnnOutputDim: Int = 256)

    public func make(observationSpace: any Space, normalizeImages: Bool) throws
        -> any FeaturesExtractor
    {
        let normalizedImage = normalizeImages
        if let dict = observationSpace as? Dict {
            switch self {
            case .auto:
                return CombinedExtractor(
                    observationSpace: dict,
                    featuresDim: 256,
                    normalizedImage: normalizedImage,
                    cnnOutputDim: 256
                )
            case .combined(let featuresDim, let cnnOutputDim):
                return CombinedExtractor(
                    observationSpace: dict,
                    featuresDim: featuresDim,
                    normalizedImage: normalizedImage,
                    cnnOutputDim: cnnOutputDim
                )
            case .flatten:
                return CombinedExtractor(
                    observationSpace: dict,
                    featuresDim: 256,
                    normalizedImage: normalizedImage,
                    cnnOutputDim: 256
                )
            case .natureCNN:
                throw GymnazoError.invalidFeatureExtractorConfiguration(
                    config: "natureCNN",
                    observationSpace: String(describing: type(of: observationSpace))
                )
            }
        }

        if let box = boxSpace(from: observationSpace) {
            let isImage = (box.shape?.count == 3)
            switch self {
            case .auto:
                if isImage {
                    return NatureCNN(
                        observationSpace: box,
                        featuresDim: 512,
                        normalizedImage: normalizedImage
                    )
                }
                return FlattenExtractor(featuresDim: box.shape?.reduce(1, *) ?? 1)
            case .flatten:
                return FlattenExtractor(featuresDim: box.shape?.reduce(1, *) ?? 1)
            case .natureCNN(let featuresDim):
                guard isImage else {
                    throw GymnazoError.invalidFeatureExtractorConfiguration(
                        config: "natureCNN",
                        observationSpace: String(describing: type(of: observationSpace))
                    )
                }
                return NatureCNN(
                    observationSpace: box,
                    featuresDim: featuresDim,
                    normalizedImage: normalizedImage
                )
            case .combined:
                throw GymnazoError.invalidFeatureExtractorConfiguration(
                    config: "combined",
                    observationSpace: String(describing: type(of: observationSpace))
                )
            }
        }

        return FlattenExtractor(featuresDim: observationSpace.shape?.reduce(1, *) ?? 1)
    }
}

/// Specifies the optimizer to build for a network.
public enum OptimizerConfig: Sendable, Codable, Equatable {
    case adam(beta1: Float = 0.9, beta2: Float = 0.999, eps: Float = 1e-8)

    public func make(learningRate: Float) -> Adam {
        switch self {
        case .adam(let beta1, let beta2, let eps):
            return Adam(learningRate: learningRate, betas: (beta1, beta2), eps: eps)
        }
    }
}

/// Network and sampling configuration for the ``SAC`` actor (``SACActor``).
public struct SACActorConfig: Sendable, Codable, Equatable {
    /// MLP architecture for the actor network.
    public var netArch: NetArch
    /// Feature extractor applied before the MLP.
    public var featuresExtractor: FeaturesExtractorConfig
    /// Activation function used between MLP layers.
    public var activation: ActivationConfig
    /// Whether to use State-Dependent Exploration.
    public var useSDE: Bool
    /// Initial value for the log-standard-deviation parameter.
    public var logStdInit: Float
    /// Whether to use a full-covariance SDE noise matrix.
    public var fullStd: Bool
    /// Clipping bound applied to the mean action output.
    public var clipMean: Float
    /// Whether to normalize uint8 observations to `[0, 1]`.
    public var normalizeImages: Bool

    public init(
        netArch: NetArch = .shared([256, 256]),
        featuresExtractor: FeaturesExtractorConfig = .auto,
        activation: ActivationConfig = .relu,
        useSDE: Bool = false,
        logStdInit: Float = -3.0,
        fullStd: Bool = true,
        clipMean: Float = 2.0,
        normalizeImages: Bool = true
    ) {
        self.netArch = netArch
        self.featuresExtractor = featuresExtractor
        self.activation = activation
        self.useSDE = useSDE
        self.logStdInit = logStdInit
        self.fullStd = fullStd
        self.clipMean = clipMean
        self.normalizeImages = normalizeImages
    }
}

/// Network configuration for the ``SAC`` critic (``SACCritic``).
public struct SACCriticConfig: Sendable, Codable, Equatable {
    /// Optional MLP hidden-layer sizes. If `nil`, uses `[256, 256]`.
    public var netArch: [Int]?
    /// Number of Q-networks to use for the ensemble (default: 2).
    public var nCritics: Int
    /// Whether the critic shares a feature extractor with the actor.
    public var shareFeaturesExtractor: Bool
    /// Optional feature extractor override; inherits from the actor config when `nil`.
    public var featuresExtractor: FeaturesExtractorConfig?
    /// Whether to normalize uint8 observations. Inherits from actor config when `nil`.
    public var normalizeImages: Bool?
    /// Activation function override. Inherits from actor config when `nil`.
    public var activation: ActivationConfig?

    public init(
        netArch: [Int]? = nil,
        nCritics: Int = 2,
        shareFeaturesExtractor: Bool = false,
        featuresExtractor: FeaturesExtractorConfig? = nil,
        normalizeImages: Bool? = nil,
        activation: ActivationConfig? = nil
    ) {
        self.netArch = netArch
        self.nCritics = nCritics
        self.shareFeaturesExtractor = shareFeaturesExtractor
        self.featuresExtractor = featuresExtractor
        self.normalizeImages = normalizeImages
        self.activation = activation
    }
}

/// Combined actor and critic network configuration for ``SAC``.
public struct SACNetworksConfig: Sendable, Codable, Equatable {
    /// Actor network configuration.
    public var actor: SACActorConfig
    /// Critic network configuration.
    public var critic: SACCriticConfig

    public init(
        actor: SACActorConfig = SACActorConfig(), critic: SACCriticConfig = SACCriticConfig()
    ) {
        self.actor = actor
        self.critic = critic
    }
}

/// Per-network optimizer configuration for ``SAC``.
public struct SACOptimizerConfig: Sendable, Codable, Equatable {
    /// Optimizer for the actor network.
    public var actor: OptimizerConfig
    /// Optimizer for the critic network.
    public var critic: OptimizerConfig
    /// Optional optimizer for the entropy coefficient. `nil` disables auto-tuning.
    public var entropy: OptimizerConfig?

    public init(
        actor: OptimizerConfig = .adam(),
        critic: OptimizerConfig = .adam(),
        entropy: OptimizerConfig? = .adam()
    ) {
        self.actor = actor
        self.critic = critic
        self.entropy = entropy
    }
}

/// The entropy regularization coefficient for ``SAC``.
///
/// - `auto`: Automatically tunes the coefficient to match a target entropy.
/// - `fixed`: Uses a constant coefficient value.
public enum EntropyCoef: Sendable, Codable, Equatable {
    case auto(init: Float = 1.0)
    case fixed(Float)

    public var isAuto: Bool {
        if case .auto = self { return true }
        return false
    }

    public var initialValue: Float {
        switch self {
        case .auto(let v): return v
        case .fixed(let v): return v
        }
    }
}
