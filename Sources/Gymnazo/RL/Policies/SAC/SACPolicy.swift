//
//  SACPolicy.swift
//  Gymnazo
//

import MLX
import MLXNN

private let logStdMax: Float = 2.0
private let logStdMin: Float = -20.0

/// Actor (Policy) Network for SAC
public final class SACActor: Module, Policy {
    public let observationSpace: any Space<MLXArray>
    public let actionSpace: any Space<MLXArray>
    public let netArch: NetArch

    public let featuresExtractor: any FeaturesExtractor
    public let normalizeImages: Bool

    private let actionDim: Int
    public let useSDE: Bool
    private let fullStd: Bool
    private let clipMean: Float

    @ModuleInfo private var latentPi: Sequential
    @ModuleInfo private var mu: Linear
    @ModuleInfo private var logStdLayer: Linear
    @ModuleInfo private var sdeLogStd: MLXArray

    private var squashedDiagGaussian = SquashedDiagGaussianDistribution()
    private var sdeDist: StateDependentNoiseDistribution

    public init(
        observationSpace: any Space<MLXArray>,
        actionSpace: any Space<MLXArray>,
        netArch: NetArch = .shared([256, 256]),
        featuresExtractor: (any FeaturesExtractor)? = nil,
        normalizeImages: Bool = true,
        activation: () -> any UnaryLayer = { ReLU() },
        useSDE: Bool = false,
        logStdInit: Float = -3.0,
        fullStd: Bool = true,
        clipMean: Float = 2.0
    ) {
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        self.netArch = netArch
        self.featuresExtractor =
            featuresExtractor
            ?? FeaturesExtractorConfig.auto.make(
                observationSpace: observationSpace,
                normalizeImages: normalizeImages
            )
        self.normalizeImages = normalizeImages
        self.useSDE = useSDE
        self.fullStd = fullStd
        self.clipMean = clipMean

        self.actionDim = getActionDim(actionSpace)

        let featureDim = self.featuresExtractor.featuresDim
        let hidden = netArch.actor
        let lastLayerDim = hidden.last ?? featureDim

        self.latentPi = MLPFactory.make(
            inputDim: featureDim,
            outputDim: 0,
            hiddenLayers: hidden,
            activation: activation
        )

        self.mu = Linear(lastLayerDim, actionDim)

        self.logStdLayer = Linear(lastLayerDim, actionDim)

        self.sdeDist = StateDependentNoiseDistribution(
            actionDim: actionDim,
            fullStd: fullStd,
            squashOutput: true,
            learnFeatures: true,
            latentSDEDim: lastLayerDim
        )

        let logStdShape: [Int] = fullStd ? [lastLayerDim, actionDim] : [lastLayerDim, 1]
        self.sdeLogStd = MLX.zeros(logStdShape) + logStdInit

        super.init()

        /// SB3 parity: initialize `log_std` head bias to `log_std_init`.
        ///
        /// SB3 sets the bias of the `log_std` linear layer so initial exploration matches the
        /// configured `log_std_init` for the diagonal Gaussian policy.
        let bias = MLX.zeros([actionDim]) + logStdInit
        _ = try? logStdLayer.update(
            parameters: ModuleParameters.unflattened(["bias": bias]),
            verify: .none
        )
    }

    public convenience init(
        observationSpace: any Space<MLXArray>,
        actionSpace: any Space<MLXArray>,
        config: SACActorConfig = SACActorConfig()
    ) {
        let extractor = config.featuresExtractor.make(
            observationSpace: observationSpace,
            normalizeImages: config.normalizeImages
        )

        self.init(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            netArch: config.netArch,
            featuresExtractor: extractor,
            normalizeImages: config.normalizeImages,
            activation: { config.activation.make() },
            useSDE: config.useSDE,
            logStdInit: config.logStdInit,
            fullStd: config.fullStd,
            clipMean: config.clipMean
        )
    }

    public func makeFeatureExtractor() -> any FeaturesExtractor {
        FlattenExtractor(featuresDim: observationSpace.shape?.reduce(1, *) ?? 1)
    }

    public var squashOutput: Bool { true }

    public func predictInternal(observation: MLXArray, deterministic: Bool) -> MLXArray {
        let (actions, _) = actionLogProbFromFeatures(observation, deterministic: deterministic)
        return actions
    }

    /// Resets exploration noise for SDE.
    ///
    /// - Parameters:
    ///   - batchSize: Number of environments.
    ///   - key: Optional RNG key for reproducible noise sampling.
    public func resetNoise(batchSize: Int = 1, key: MLXArray? = nil) {
        guard useSDE else { return }
        sdeDist.sampleWeights(logStd: sdeLogStd, batchSize: batchSize, key: key)
    }

    public func getStd() -> MLXArray? {
        guard useSDE else { return nil }
        return sdeDist.getStd(logStd: sdeLogStd)
    }

    /// Computes action and log probability from observation.
    ///
    /// - Parameters:
    ///   - obs: Observation tensor.
    ///   - deterministic: If true, returns the mode; otherwise samples.
    ///   - key: Optional RNG key for reproducible sampling.
    /// - Returns: Tuple of (actions, log_prob).
    public func actionLogProb(
        obs: MLXArray,
        deterministic: Bool = false,
        key: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let features = extractFeatures(obs: obs, featuresExtractor: featuresExtractor)
        return actionLogProbFromFeatures(features, deterministic: deterministic, key: key)
    }

    /// Computes action and log probability from dictionary observation.
    ///
    /// - Parameters:
    ///   - obs: Dictionary observation.
    ///   - deterministic: If true, returns the mode; otherwise samples.
    ///   - key: Optional RNG key for reproducible sampling.
    /// - Returns: Tuple of (actions, log_prob).
    public func actionLogProb(
        obs: [String: MLXArray],
        deterministic: Bool = false,
        key: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        guard let dictExtractor = featuresExtractor as? any DictFeaturesExtractor else {
            preconditionFailure(
                "actionLogProb(obs: [String: MLXArray]) requires a DictFeaturesExtractor")
        }
        let features = extractFeatures(obs: obs, featuresExtractor: dictExtractor)
        return actionLogProbFromFeatures(features, deterministic: deterministic, key: key)
    }

    private func actionLogProbFromFeatures(
        _ features: MLXArray,
        deterministic: Bool,
        key: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let latent = latentPi(features)

        var meanActions = mu(latent)

        if useSDE {
            if clipMean > 0 {
                meanActions = MLX.clip(meanActions, min: -clipMean, max: clipMean)
            }
            sdeDist.probaDistribution(
                meanActions: meanActions, logStd: sdeLogStd, latentSDE: latent)
            let actions = sdeDist.getActions(deterministic: deterministic, key: key)
            let logProb = sdeDist.logProb(actions)
            return (actions, logProb)
        }

        var logStd = logStdLayer(latent)
        logStd = MLX.clip(logStd, min: logStdMin, max: logStdMax)

        squashedDiagGaussian.probaDistribution(meanActions: meanActions, logStd: logStd)
        let actions =
            deterministic ? squashedDiagGaussian.mode() : squashedDiagGaussian.sample(key: key)
        let logProb = squashedDiagGaussian.logProb(actions)
        return (actions, logProb)
    }
}
