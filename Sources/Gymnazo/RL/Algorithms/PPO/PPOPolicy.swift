import MLX
import MLXNN

/// The actor-critic neural network policy used by ``PPO``.
///
/// `PPOPolicy` owns the MLP feature extractor, action distribution, and value head.
/// It supports discrete, continuous (Gaussian / SDE), multi-discrete, and multi-binary
/// action spaces.
public final class PPOPolicy: Module, ActorCriticPolicy, @unchecked Sendable {
    public let observationSpace: any Space
    public let actionSpace: any Space
    public let netArch: NetArch
    public let normalizeImages: Bool
    public let shareFeatureExtractor: Bool
    public let featuresDim: Int
    public let featuresExtractor: any FeaturesExtractor
    public let piFeatureExtractor: any FeaturesExtractor
    public let vfFeatureExtractor: any FeaturesExtractor

    private let activation: ActivationConfig
    private let featuresExtractorConfig: FeaturesExtractorConfig
    private let orthoInitValue: Bool
    private let useSDEValue: Bool
    private let logStdInitValue: Float
    private let fullStdValue: Bool
    private let actionSpaceKind: ActionSpaceKind

    @ModuleInfo private var mlpExtractorModule: MLPExtractor
    @ModuleInfo private var actionLinear: Linear
    @ModuleInfo private var valueLinear: Linear
    @ModuleInfo private var diagLogStd: MLXArray
    @ModuleInfo private var sdeLogStd: MLXArray

    public let actionDist: any Distribution

    public var orthoInit: Bool { orthoInitValue }
    public var useSDE: Bool { useSDEValue }
    public var logStdInit: Float { logStdInitValue }
    public var fullStd: Bool { fullStdValue }
    public var mlpExtractor: MLPExtractor { mlpExtractorModule }
    public var actionNet: any UnaryLayer { actionLinear }
    public var valueNet: Linear { valueLinear }

    public var logStd: MLXArray? {
        get {
            guard case .box = actionSpaceKind else { return nil }
            return useSDE ? sdeLogStd : diagLogStd
        }
        set {
            guard case .box = actionSpaceKind, let newValue else { return }
            if useSDE {
                sdeLogStd = newValue
            } else {
                diagLogStd = newValue
            }
        }
    }

    /// Creates a `PPOPolicy` with fine-grained control over all options.
    ///
    /// - Parameters:
    ///   - observationSpace: The environment observation space.
    ///   - actionSpace: The environment action space.
    ///   - netArch: MLP architecture for actor and critic. Defaults to `[64, 64]`.
    ///   - featuresExtractor: Feature extractor configuration.
    ///   - activation: Activation function used between MLP layers.
    ///   - normalizeImages: Whether to normalize uint8 observations to `[0, 1]`.
    ///   - shareFeatureExtractor: Whether actor and critic share the same extractor.
    ///   - orthoInit: Whether to apply orthogonal weight initialization.
    ///   - useSDE: Whether to use State-Dependent Exploration.
    ///   - logStdInit: Initial log-std for continuous distributions.
    ///   - fullStd: Whether to use a full-covariance SDE noise matrix.
    public init(
        observationSpace: any Space,
        actionSpace: any Space,
        netArch: NetArch = .shared([64, 64]),
        featuresExtractor: FeaturesExtractorConfig = .auto,
        activation: ActivationConfig = .tanh,
        normalizeImages: Bool = true,
        shareFeatureExtractor: Bool = true,
        orthoInit: Bool = true,
        useSDE: Bool = false,
        logStdInit: Float = 0.0,
        fullStd: Bool = true
    ) throws {
        let resolvedKind = try Self.makeActionSpaceKind(actionSpace)
        if useSDE {
            if case .box = resolvedKind {
            } else {
                throw GymnazoError.invalidConfiguration(
                    "PPOPolicy useSDE requires a Box action space."
                )
            }
        }

        let extractors = try Self.makeFeatureExtractors(
            observationSpace: observationSpace,
            config: featuresExtractor,
            normalizeImages: normalizeImages,
            shareFeatureExtractor: shareFeatureExtractor
        )
        let piFeaturesDim = extractors.pi.featuresDim
        let vfFeaturesDim = extractors.vf.featuresDim
        guard piFeaturesDim == vfFeaturesDim else {
            throw GymnazoError.invalidConfiguration(
                "PPOPolicy requires equal actor/critic feature dimensions."
            )
        }

        let mlp = MLPExtractor(
            featureDim: piFeaturesDim,
            netArch: netArch,
            activation: { activation.make() }
        )
        let actionOutDim = resolvedKind.actionOutputDim
        let actionLayer = Linear(mlp.latentDimPi, actionOutDim)
        let valueLayer = Linear(mlp.latentDimVf, 1)

        let diagLogStdShape: [Int]
        let sdeLogStdShape: [Int]
        switch resolvedKind {
        case .box(let actionDim):
            diagLogStdShape = useSDE ? [1] : [actionDim]
            if useSDE {
                sdeLogStdShape = fullStd
                    ? [mlp.latentDimPi, actionDim]
                    : [mlp.latentDimPi, 1]
            } else {
                sdeLogStdShape = [1, 1]
            }
        default:
            diagLogStdShape = [1]
            sdeLogStdShape = [1, 1]
        }
        let diagLogStdParam = MLX.zeros(diagLogStdShape) + logStdInit
        let sdeLogStdParam = MLX.zeros(sdeLogStdShape) + logStdInit

        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        self.netArch = netArch
        self.normalizeImages = normalizeImages
        self.shareFeatureExtractor = shareFeatureExtractor
        self.featuresExtractor = extractors.shared
        self.piFeatureExtractor = extractors.pi
        self.vfFeatureExtractor = extractors.vf
        self.featuresDim = piFeaturesDim
        self.activation = activation
        self.featuresExtractorConfig = featuresExtractor
        self.orthoInitValue = orthoInit
        self.useSDEValue = useSDE
        self.logStdInitValue = logStdInit
        self.fullStdValue = fullStd
        self.actionSpaceKind = resolvedKind
        self.mlpExtractorModule = mlp
        self.actionLinear = actionLayer
        self.valueLinear = valueLayer
        self.diagLogStd = diagLogStdParam
        self.sdeLogStd = sdeLogStdParam
        self.actionDist = Self.makeDistribution(
            kind: resolvedKind,
            useSDE: useSDE,
            fullStd: fullStd,
            latentSDEDim: mlp.latentDimPi
        )

        super.init()

        if orthoInit {
            if shareFeatureExtractor {
                try applyActorCriticOrthoInit(
                    featuresExtractor: self.featuresExtractor,
                    mlpExtractor: mlpExtractorModule,
                    actionNet: actionLinear,
                    valueNet: valueLinear,
                    shareFeatureExtractor: true
                )
            } else {
                try applyActorCriticOrthoInit(
                    featuresExtractor: self.piFeatureExtractor,
                    mlpExtractor: mlpExtractorModule,
                    actionNet: actionLinear,
                    valueNet: valueLinear,
                    shareFeatureExtractor: false,
                    piFeatureExtractor: self.piFeatureExtractor,
                    vfFeatureExtractor: self.vfFeatureExtractor
                )
            }
        }
    }

    public convenience init(
        observationSpace: any Space,
        actionSpace: any Space,
        config: PPOPolicyConfig,
        useSDE: Bool
    ) throws {
        try self.init(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            netArch: config.netArch,
            featuresExtractor: config.featuresExtractor,
            activation: config.activation,
            normalizeImages: config.normalizeImages,
            shareFeatureExtractor: config.shareFeaturesExtractor,
            orthoInit: config.orthoInit,
            useSDE: useSDE,
            logStdInit: config.logStdInit,
            fullStd: config.fullStd
        )
    }

    public func makeFeatureExtractor() -> any FeaturesExtractor {
        (try? featuresExtractorConfig.make(
            observationSpace: observationSpace,
            normalizeImages: normalizeImages
        )) ?? FlattenExtractor(featuresDim: observationSpace.shape?.reduce(1, *) ?? 1)
    }

    /// Runs a forward pass and returns actions, values, and log-probabilities.
    ///
    /// - Parameters:
    ///   - obs: The current observation.
    ///   - deterministic: If `true`, returns the mode action.
    /// - Returns: A tuple of `(actions, values, logProbs)`.
    public func callAsFunction(obs: MLXArray, deterministic: Bool) -> (MLXArray, MLXArray, MLXArray) {
        let output = forward(obs: obs, deterministic: deterministic, key: nil)
        return (output.actions, output.values, output.logProb)
    }

    /// Evaluates the log-probability and entropy of the given actions under the current policy.
    ///
    /// - Parameters:
    ///   - obs: The observations at which actions were taken.
    ///   - actions: The actions to evaluate.
    /// - Returns: A tuple of `(values, logProbs, entropy)` where `entropy` may be `nil`.
    public func evaluateActions(obs: MLXArray, actions: MLXArray) -> (MLXArray, MLXArray, MLXArray?) {
        let (latentPi, latentVf) = latentFeatures(obs: obs)
        let distribution = prepareDistribution(latentPi: latentPi)
        let preparedActions = prepareActionsForDistribution(actions)
        let values = valueLinear(latentVf)
        let logProb = distribution.logProb(preparedActions)
        let entropy = distribution.entropy()
        return (values, logProb, entropy)
    }

    /// Returns the action distribution conditioned on the given observation.
    ///
    /// - Parameter obs: The current observation.
    /// - Returns: The parameterized action distribution.
    public func getDistribution(obs: MLXArray) -> any Distribution {
        let (latentPi, _) = latentFeatures(obs: obs)
        return prepareDistribution(latentPi: latentPi)
    }

    /// Predicts the estimated state value for the given observation without computing actions.
    ///
    /// - Parameter obs: The current observation.
    /// - Returns: The estimated state value.
    public func predictValues(obs: MLXArray) -> MLXArray {
        let (_, latentVf) = latentFeatures(obs: obs)
        return valueLinear(latentVf)
    }

    /// Resets the SDE exploration noise using an internally generated key.
    ///
    /// - Parameter nEnvs: Number of parallel environments.
    public func resetNoise(nEnvs: Int) {
        resetNoise(nEnvs: nEnvs, key: nil)
    }

    /// Resets the SDE exploration noise using the provided key.
    ///
    /// - Parameters:
    ///   - nEnvs: Number of parallel environments.
    ///   - key: Optional PRNG key. Passes `nil` to use a random key.
    public func resetNoise(nEnvs: Int, key: MLXArray?) {
        guard useSDE, let sde = actionDist as? StateDependentNoiseDistribution else {
            return
        }
        sde.sampleWeights(logStd: sdeLogStd, batchSize: nEnvs, key: key)
    }

    public func forward(
        obs: MLXArray,
        deterministic: Bool = false,
        key: MLXArray? = nil
    ) -> ActorCriticOutput {
        let (latentPi, latentVf) = latentFeatures(obs: obs)
        let distribution = prepareDistribution(latentPi: latentPi)
        let actions = distribution.getActions(deterministic: deterministic, key: key)
        let preparedActions = prepareActionsForDistribution(actions)
        let logProb = distribution.logProb(preparedActions)
        let values = valueLinear(latentVf)
        return ActorCriticOutput(actions: actions, values: values, logProb: logProb)
    }

    private func latentFeatures(obs: MLXArray) -> (MLXArray, MLXArray) {
        let (piFeatures, vfFeatures) = extractActorCriticFeatures(obs: obs)
        if shareFeatureExtractor {
            return mlpExtractorModule(piFeatures)
        } else {
            return (
                mlpExtractorModule.actor(piFeatures),
                mlpExtractorModule.critic(vfFeatures)
            )
        }
    }

    private func prepareDistribution(latentPi: MLXArray) -> any Distribution {
        switch actionSpaceKind {
        case .box:
            let meanActions = actionLinear(latentPi)
            if useSDE {
                guard let sde = actionDist as? StateDependentNoiseDistribution else {
                    preconditionFailure("Expected StateDependentNoiseDistribution.")
                }
                sde.probaDistribution(
                    meanActions: meanActions,
                    logStd: sdeLogStd,
                    latentSDE: latentPi
                )
                return sde
            }
            guard let gaussian = actionDist as? DiagGaussianDistribution else {
                preconditionFailure("Expected DiagGaussianDistribution.")
            }
            gaussian.probaDistribution(meanActions: meanActions, logStd: diagLogStd)
            return gaussian
        case .discrete:
            guard let categorical = actionDist as? CategoricalDistribution else {
                preconditionFailure("Expected CategoricalDistribution.")
            }
            categorical.probaDistribution(actionLogits: actionLinear(latentPi))
            return categorical
        case .multiDiscrete:
            guard let multiCategorical = actionDist as? MultiCategoricalDistribution else {
                preconditionFailure("Expected MultiCategoricalDistribution.")
            }
            multiCategorical.probaDistribution(actionLogits: actionLinear(latentPi))
            return multiCategorical
        case .multiBinary:
            guard let bernoulli = actionDist as? BernoulliDistribution else {
                preconditionFailure("Expected BernoulliDistribution.")
            }
            bernoulli.probaDistribution(actionLogits: actionLinear(latentPi))
            return bernoulli
        }
    }

    private func prepareActionsForDistribution(_ actions: MLXArray) -> MLXArray {
        switch actionSpaceKind {
        case .box:
            return actions.asType(.float32)
        case .discrete:
            var prepared = actions
            if prepared.ndim > 1 {
                prepared = prepared.reshaped([-1])
            }
            return prepared.asType(.int32)
        case .multiDiscrete:
            var prepared = actions
            if prepared.ndim == 1 {
                prepared = prepared.reshaped([1, prepared.shape[0]])
            }
            return prepared.asType(.int32)
        case .multiBinary:
            var prepared = actions
            if prepared.ndim == 1 {
                prepared = prepared.reshaped([1, prepared.shape[0]])
            }
            return prepared.asType(.float32)
        }
    }

    private static func makeFeatureExtractors(
        observationSpace: any Space,
        config: FeaturesExtractorConfig,
        normalizeImages: Bool,
        shareFeatureExtractor: Bool
    ) throws -> (shared: any FeaturesExtractor, pi: any FeaturesExtractor, vf: any FeaturesExtractor) {
        let shared = try config.make(
            observationSpace: observationSpace,
            normalizeImages: normalizeImages
        )
        if shareFeatureExtractor {
            return (shared, shared, shared)
        }
        let pi = try config.make(
            observationSpace: observationSpace,
            normalizeImages: normalizeImages
        )
        let vf = try config.make(
            observationSpace: observationSpace,
            normalizeImages: normalizeImages
        )
        return (shared, pi, vf)
    }

    private static func makeDistribution(
        kind: ActionSpaceKind,
        useSDE: Bool,
        fullStd: Bool,
        latentSDEDim: Int
    ) -> any Distribution {
        switch kind {
        case .box(let actionDim):
            if useSDE {
                return StateDependentNoiseDistribution(
                    actionDim: actionDim,
                    fullStd: fullStd,
                    squashOutput: false,
                    learnFeatures: false,
                    latentSDEDim: latentSDEDim
                )
            }
            return DiagGaussianDistribution(actionDim: actionDim)
        case .discrete(let actionDim):
            return CategoricalDistribution(actionDim: actionDim)
        case .multiDiscrete(let actionDims, _):
            return MultiCategoricalDistribution(actionDims: actionDims)
        case .multiBinary(let actionDim):
            return BernoulliDistribution(actionDim: actionDim)
        }
    }

    private static func makeActionSpaceKind(_ actionSpace: any Space) throws -> ActionSpaceKind {
        if let box = boxSpace(from: actionSpace) {
            return .box(actionDim: box.shape?.reduce(1, *) ?? 1)
        }
        if let discrete = actionSpace as? Discrete {
            return .discrete(actionDim: discrete.n)
        }
        if let multiDiscrete = actionSpace as? MultiDiscrete {
            let dims = multiDiscrete.nvec.asType(.int32).reshaped([-1]).asArray(Int32.self).map(Int.init)
            return .multiDiscrete(actionDims: dims, actionDim: dims.reduce(0, +))
        }
        if let multiBinary = actionSpace as? MultiBinary {
            return .multiBinary(actionDim: multiBinary.shape?.reduce(1, *) ?? 1)
        }
        throw GymnazoError.invalidActionType(
            expected: "Box | Discrete | MultiDiscrete | MultiBinary",
            actual: String(describing: type(of: actionSpace))
        )
    }
}

private enum ActionSpaceKind {
    case box(actionDim: Int)
    case discrete(actionDim: Int)
    case multiDiscrete(actionDims: [Int], actionDim: Int)
    case multiBinary(actionDim: Int)

    var actionOutputDim: Int {
        switch self {
        case .box(let actionDim):
            return actionDim
        case .discrete(let actionDim):
            return actionDim
        case .multiDiscrete(_, let actionDim):
            return actionDim
        case .multiBinary(let actionDim):
            return actionDim
        }
    }
}
