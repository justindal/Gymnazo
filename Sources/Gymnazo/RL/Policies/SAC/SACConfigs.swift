import MLX
import MLXNN
import MLXOptimizers

public enum ActivationConfig: String, Sendable, Codable {
    case relu

    public func make() -> any UnaryLayer {
        switch self {
        case .relu:
            return ReLU()
        }
    }
}

public enum FeaturesExtractorConfig: Sendable, Codable, Equatable {
    case auto
    case flatten
    case natureCNN(featuresDim: Int = 512)
    case combined(featuresDim: Int = 256, cnnOutputDim: Int = 256)

    public func make(observationSpace: any Space, normalizeImages: Bool) -> any FeaturesExtractor {
        if let dict = observationSpace as? Dict {
            switch self {
            case .auto:
                return CombinedExtractor(
                    observationSpace: dict,
                    featuresDim: 256,
                    normalizedImage: normalizeImages,
                    cnnOutputDim: 256
                )
            case .combined(let featuresDim, let cnnOutputDim):
                return CombinedExtractor(
                    observationSpace: dict,
                    featuresDim: featuresDim,
                    normalizedImage: normalizeImages,
                    cnnOutputDim: cnnOutputDim
                )
            case .flatten:
                return CombinedExtractor(
                    observationSpace: dict,
                    featuresDim: 256,
                    normalizedImage: normalizeImages,
                    cnnOutputDim: 256
                )
            case .natureCNN:
                preconditionFailure("NatureCNN requires a Box observation space, got Dict.")
            }
        }

        if let box = observationSpace as? Box {
            let isImage = (box.shape?.count == 3)
            switch self {
            case .auto:
                if isImage {
                    return NatureCNN(
                        observationSpace: box,
                        featuresDim: 512,
                        normalizedImage: normalizeImages
                    )
                }
                return FlattenExtractor(featuresDim: box.shape?.reduce(1, *) ?? 1)
            case .flatten:
                return FlattenExtractor(featuresDim: box.shape?.reduce(1, *) ?? 1)
            case .natureCNN(let featuresDim):
                precondition(
                    isImage, "NatureCNN requires a Box observation space with 3 dims [C,H,W].")
                return NatureCNN(
                    observationSpace: box,
                    featuresDim: featuresDim,
                    normalizedImage: normalizeImages
                )
            case .combined:
                preconditionFailure("CombinedExtractor requires a Dict observation space, got Box.")
            }
        }

        return FlattenExtractor(featuresDim: observationSpace.shape?.reduce(1, *) ?? 1)
    }
}

public enum OptimizerConfig: Sendable, Codable, Equatable {
    case adam(beta1: Float = 0.9, beta2: Float = 0.999, eps: Float = 1e-8)

    public func make(learningRate: Float) -> Adam {
        switch self {
        case .adam(let beta1, let beta2, let eps):
            return Adam(learningRate: learningRate, betas: (beta1, beta2), eps: eps)
        }
    }
}

public struct SACActorConfig: Sendable, Codable, Equatable {
    public var netArch: NetArch
    public var featuresExtractor: FeaturesExtractorConfig
    public var activation: ActivationConfig
    public var useSDE: Bool
    public var logStdInit: Float
    public var fullStd: Bool
    public var clipMean: Float
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

public struct SACCriticConfig: Sendable, Codable, Equatable {
    public var netArch: [Int]?
    public var nCritics: Int
    public var shareFeaturesExtractor: Bool
    public var featuresExtractor: FeaturesExtractorConfig?
    public var normalizeImages: Bool?
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

public struct SACNetworksConfig: Sendable, Codable, Equatable {
    public var actor: SACActorConfig
    public var critic: SACCriticConfig

    public init(
        actor: SACActorConfig = SACActorConfig(), critic: SACCriticConfig = SACCriticConfig()
    ) {
        self.actor = actor
        self.critic = critic
    }
}

public struct SACOptimizerConfig: Sendable, Codable, Equatable {
    public var actor: OptimizerConfig
    public var critic: OptimizerConfig
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
