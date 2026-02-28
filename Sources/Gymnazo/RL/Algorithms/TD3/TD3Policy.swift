import MLX
import MLXNN
import MLXOptimizers

/// The combined actor-critic policy for ``TD3``.
///
/// Owns the actor, target actor, critic, and target critic networks, along with their
/// optimizers. Target networks are initialised as copies of the online networks.
public final class TD3Policy: Module, Policy, @unchecked Sendable {
    public let observationSpace: any Space
    public let actionSpace: any Space
    public let learningRateSchedule: any LearningRateSchedule
    public let netArch: NetArch
    public let activation: ActivationConfig
    public let featuresExtractorConfig: FeaturesExtractorConfig
    public let normalizeImages: Bool
    public let nCritics: Int
    public let shareFeaturesExtractor: Bool
    public let actorOptimizerConfig: OptimizerConfig
    public let criticOptimizerConfig: OptimizerConfig

    @ModuleInfo public var actor: TD3Actor
    @ModuleInfo public var actorTarget: TD3Actor
    @ModuleInfo public var critic: SACCritic
    @ModuleInfo public var criticTarget: SACCritic

    public var actorOptimizer: Adam
    public var criticOptimizer: Adam

    public var featuresExtractor: any FeaturesExtractor { actor.featuresExtractor }
    public var squashOutput: Bool { true }

    public init(
        observationSpace: any Space,
        actionSpace: any Space,
        learningRateSchedule: any LearningRateSchedule = ConstantLearningRate(1e-3),
        netArch: NetArch? = nil,
        activation: ActivationConfig = .relu,
        featuresExtractor: FeaturesExtractorConfig = .auto,
        normalizeImages: Bool = true,
        nCritics: Int = 2,
        shareFeaturesExtractor: Bool = false,
        actorOptimizer: OptimizerConfig = .adam(),
        criticOptimizer: OptimizerConfig = .adam()
    ) {
        let resolvedNetArch =
            netArch
            ?? Self.defaultNetArch(
                observationSpace: observationSpace,
                featuresExtractor: featuresExtractor
            )
        let resolvedNCritics = max(1, nCritics)
        let setup = Self.buildSetup(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            netArch: resolvedNetArch,
            activation: activation,
            featuresExtractor: featuresExtractor,
            normalizeImages: normalizeImages,
            nCritics: resolvedNCritics,
            shareFeaturesExtractor: shareFeaturesExtractor,
            actorOptimizerConfig: actorOptimizer,
            criticOptimizerConfig: criticOptimizer,
            learningRateSchedule: learningRateSchedule
        )

        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        self.learningRateSchedule = learningRateSchedule
        self.netArch = resolvedNetArch
        self.activation = activation
        self.featuresExtractorConfig = featuresExtractor
        self.normalizeImages = normalizeImages
        self.nCritics = resolvedNCritics
        self.shareFeaturesExtractor = shareFeaturesExtractor
        self.actorOptimizerConfig = actorOptimizer
        self.criticOptimizerConfig = criticOptimizer
        self.actor = setup.actor
        self.actorTarget = setup.actorTarget
        self.critic = setup.critic
        self.criticTarget = setup.criticTarget
        self.actorOptimizer = setup.actorOptimizer
        self.criticOptimizer = setup.criticOptimizer

        super.init()

        syncTargetsFromOnline()
    }

    public convenience init(
        observationSpace: any Space,
        actionSpace: any Space,
        learningRateSchedule: any LearningRateSchedule = ConstantLearningRate(1e-3),
        config: TD3PolicyConfig = TD3PolicyConfig()
    ) {
        self.init(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            learningRateSchedule: learningRateSchedule,
            netArch: config.netArch,
            activation: config.activation,
            featuresExtractor: config.featuresExtractor,
            normalizeImages: config.normalizeImages,
            nCritics: config.nCritics,
            shareFeaturesExtractor: config.shareFeaturesExtractor,
            actorOptimizer: config.actorOptimizer,
            criticOptimizer: config.criticOptimizer
        )
    }

    public func callAsFunction(_ observation: MLXArray, deterministic: Bool) -> MLXArray {
        actor(observation, deterministic: deterministic)
    }

    public func predict(observation: MLXArray, deterministic: Bool = true) -> MLXArray {
        actor.predict(observation: observation, deterministic: deterministic)
    }

    public func predict(observation: [String: MLXArray], deterministic: Bool = true) -> MLXArray {
        actor.predict(observation: observation, deterministic: deterministic)
    }

    public func makeFeatureExtractor() -> any FeaturesExtractor {
        featuresExtractorConfig.make(
            observationSpace: observationSpace,
            normalizeImages: normalizeImages
        )
    }

    public func setTrainingMode(_ mode: Bool) {
        actor.train(mode)
        critic.train(mode)
        actorTarget.train(false)
        criticTarget.train(false)
    }

    public func updateOptimizers(progressRemaining: Double) {
        let lr = Float(learningRateSchedule.value(at: progressRemaining))
        actorOptimizer.learningRate = lr
        criticOptimizer.learningRate = lr
    }

    public func syncTargetsFromOnline() {
        _ = try? actorTarget.update(parameters: actor.parameters(), verify: .noUnusedKeys)
        _ = try? criticTarget.update(parameters: critic.parameters(), verify: .noUnusedKeys)
        actorTarget.train(false)
        criticTarget.train(false)
    }

    private static func defaultNetArch(
        observationSpace: any Space,
        featuresExtractor: FeaturesExtractorConfig
    ) -> NetArch {
        let useCnnDefault: Bool
        switch featuresExtractor {
        case .natureCNN:
            useCnnDefault = true
        case .auto:
            if let box = boxSpace(from: observationSpace), let shape = box.shape {
                useCnnDefault = shape.count == 3
            } else {
                useCnnDefault = false
            }
        default:
            useCnnDefault = false
        }

        return .shared(useCnnDefault ? [256, 256] : [400, 300])
    }

    private struct PolicySetup {
        let actor: TD3Actor
        let actorTarget: TD3Actor
        let critic: SACCritic
        let criticTarget: SACCritic
        let actorOptimizer: Adam
        let criticOptimizer: Adam
    }

    private static func buildSetup(
        observationSpace: any Space,
        actionSpace: any Space,
        netArch: NetArch,
        activation: ActivationConfig,
        featuresExtractor: FeaturesExtractorConfig,
        normalizeImages: Bool,
        nCritics: Int,
        shareFeaturesExtractor: Bool,
        actorOptimizerConfig: OptimizerConfig,
        criticOptimizerConfig: OptimizerConfig,
        learningRateSchedule: any LearningRateSchedule
    ) -> PolicySetup {
        let actor = TD3Actor(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            netArch: .shared(netArch.actor),
            featuresExtractor: featuresExtractor.make(
                observationSpace: observationSpace,
                normalizeImages: normalizeImages
            ),
            normalizeImages: normalizeImages,
            activation: { activation.make() }
        )

        let actorTarget = TD3Actor(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            netArch: .shared(netArch.actor),
            featuresExtractor: featuresExtractor.make(
                observationSpace: observationSpace,
                normalizeImages: normalizeImages
            ),
            normalizeImages: normalizeImages,
            activation: { activation.make() }
        )

        let criticExtractor: (any FeaturesExtractor)? =
            shareFeaturesExtractor
            ? actor.featuresExtractor
            : featuresExtractor.make(
                observationSpace: observationSpace,
                normalizeImages: normalizeImages
            )

        let critic = SACCritic(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            normalizeImages: normalizeImages,
            netArch: netArch.critic,
            featuresExtractor: criticExtractor,
            nCritics: nCritics,
            shareFeaturesExtractor: shareFeaturesExtractor,
            activation: { activation.make() }
        )

        let criticTargetExtractor: (any FeaturesExtractor)? =
            shareFeaturesExtractor
            ? actorTarget.featuresExtractor
            : featuresExtractor.make(
                observationSpace: observationSpace,
                normalizeImages: normalizeImages
            )

        let criticTarget = SACCritic(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            normalizeImages: normalizeImages,
            netArch: netArch.critic,
            featuresExtractor: criticTargetExtractor,
            nCritics: nCritics,
            shareFeaturesExtractor: shareFeaturesExtractor,
            activation: { activation.make() }
        )

        let lr = Float(learningRateSchedule.value(at: 1.0))
        let actorOptimizer = actorOptimizerConfig.make(learningRate: lr)
        let criticOptimizer = criticOptimizerConfig.make(learningRate: lr)

        return PolicySetup(
            actor: actor,
            actorTarget: actorTarget,
            critic: critic,
            criticTarget: criticTarget,
            actorOptimizer: actorOptimizer,
            criticOptimizer: criticOptimizer
        )
    }
}
