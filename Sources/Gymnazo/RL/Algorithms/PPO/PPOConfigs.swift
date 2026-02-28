import Foundation

/// Hyper-parameter configuration for ``PPO``.
public struct PPOConfig: Sendable, Codable, Equatable {
    /// Number of environment steps to collect per rollout before each update.
    public let nSteps: Int
    /// Mini-batch size used when iterating over the rollout buffer.
    public let batchSize: Int
    /// Number of optimization epochs per rollout.
    public let nEpochs: Int
    /// Discount factor. Clamped to `[0, 1]`.
    public let gamma: Double
    /// GAE lambda for advantage estimation. Clamped to `[0, 1]`.
    public let gaeLambda: Double
    /// PPO clipping coefficient for the surrogate objective.
    public let clipRange: Double
    /// Optional clipping range for value-function loss. `nil` disables clipping.
    public let clipRangeVf: Double?
    /// Whether to normalize advantages within each mini-batch.
    public let normalizeAdvantage: Bool
    /// Entropy coefficient added to the loss to encourage exploration.
    public let entCoef: Double
    /// Value-function loss coefficient.
    public let vfCoef: Double
    /// Maximum gradient norm for gradient clipping. `0` disables clipping.
    public let maxGradNorm: Double
    /// Optional KL-divergence threshold for early stopping within an epoch.
    public let targetKL: Double?
    /// Whether to use State-Dependent Exploration (SDE).
    public let useSDE: Bool
    /// Frequency at which SDE noise is re-sampled. `-1` samples only at the start of each rollout.
    public let sdeSampleFreq: Int

    public init(
        nSteps: Int = 2048,
        batchSize: Int = 64,
        nEpochs: Int = 10,
        gamma: Double = 0.99,
        gaeLambda: Double = 0.95,
        clipRange: Double = 0.2,
        clipRangeVf: Double? = nil,
        normalizeAdvantage: Bool = true,
        entCoef: Double = 0.0,
        vfCoef: Double = 0.5,
        maxGradNorm: Double = 0.5,
        targetKL: Double? = nil,
        useSDE: Bool = false,
        sdeSampleFreq: Int = -1
    ) {
        let safeNSteps = max(1, nSteps)
        var safeBatchSize = max(1, batchSize)
        var safeNormalizeAdvantage = normalizeAdvantage
        if safeNormalizeAdvantage && safeBatchSize < 2 {
            safeBatchSize = 2
        }
        safeBatchSize = min(safeBatchSize, safeNSteps)
        if safeNormalizeAdvantage && safeBatchSize < 2 {
            safeNormalizeAdvantage = false
        }

        self.nSteps = safeNSteps
        self.batchSize = safeBatchSize
        self.nEpochs = max(1, nEpochs)
        self.gamma = min(max(gamma, 0.0), 1.0)
        self.gaeLambda = min(max(gaeLambda, 0.0), 1.0)
        self.clipRange = max(0.0, clipRange)
        if let clipRangeVf {
            self.clipRangeVf = clipRangeVf > 0.0 ? clipRangeVf : nil
        } else {
            self.clipRangeVf = nil
        }
        self.normalizeAdvantage = safeNormalizeAdvantage
        self.entCoef = max(0.0, entCoef)
        self.vfCoef = max(0.0, vfCoef)
        self.maxGradNorm = max(0.0, maxGradNorm)
        if let targetKL {
            self.targetKL = targetKL > 0.0 ? targetKL : nil
        } else {
            self.targetKL = nil
        }
        self.useSDE = useSDE
        self.sdeSampleFreq = useSDE ? (sdeSampleFreq > 0 ? sdeSampleFreq : -1) : -1
    }
}

/// Network architecture and initialization configuration for ``PPOPolicy``.
public struct PPOPolicyConfig: Sendable, Codable, Equatable {
    /// MLP network architecture shared between actor and critic heads.
    public let netArch: NetArch
    /// Feature extractor applied before the MLP.
    public let featuresExtractor: FeaturesExtractorConfig
    /// Activation function used between MLP layers.
    public let activation: ActivationConfig
    /// Whether to normalize uint8 image observations to `[0, 1]`.
    public let normalizeImages: Bool
    /// Whether the actor and critic share the same feature extractor.
    public let shareFeaturesExtractor: Bool
    /// Whether to apply orthogonal initialization to the network weights.
    public let orthoInit: Bool
    /// Initial value for the log-standard-deviation parameter (continuous actions).
    public let logStdInit: Float
    /// If `true`, uses a full-covariance SDE noise matrix; otherwise uses a diagonal.
    public let fullStd: Bool

    public init(
        netArch: NetArch = .shared([64, 64]),
        featuresExtractor: FeaturesExtractorConfig = .auto,
        activation: ActivationConfig = .tanh,
        normalizeImages: Bool = true,
        shareFeaturesExtractor: Bool = true,
        orthoInit: Bool = true,
        logStdInit: Float = 0.0,
        fullStd: Bool = true
    ) {
        self.netArch = netArch
        self.featuresExtractor = featuresExtractor
        self.activation = activation
        self.normalizeImages = normalizeImages
        self.shareFeaturesExtractor = shareFeaturesExtractor
        self.orthoInit = orthoInit
        self.logStdInit = logStdInit
        self.fullStd = fullStd
    }
}
