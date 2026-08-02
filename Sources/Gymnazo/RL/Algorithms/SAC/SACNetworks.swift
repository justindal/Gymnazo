import MLX
import MLXNN

public struct SACNetworks {
    public let actor: SACActor
    public let critic: SACCritic
    public let criticTarget: SACCritic

    public init(actor: SACActor, critic: SACCritic, criticTarget: SACCritic) {
        self.actor = actor
        self.critic = critic
        self.criticTarget = criticTarget
    }

    public func syncCriticTargetFromCritic() throws {
        do {
            try criticTarget.update(parameters: critic.parameters(), verify: .noUnusedKeys)
        } catch {
            throw GymnazoError.operationFailed(
                "Failed to synchronize SAC critic target network: \(error)"
            )
        }
        criticTarget.train(false)
    }

    public init(
        observationSpace: any Space,
        actionSpace: any Space,
        config: SACNetworksConfig = SACNetworksConfig(),
        initializationSeed: UInt64? = nil
    ) throws {
        var initializationKey = initializationSeed.map(MLX.key)
        func nextInitializationKey() -> MLXArray? {
            guard let key = initializationKey else { return nil }
            let (next, updatedKey) = MLX.split(key: key, stream: .cpu)
            initializationKey = updatedKey
            return next
        }
        let actor = try SACActor(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            config: config.actor,
            initializationKey: nextInitializationKey()
        )

        let criticNetArch = config.critic.netArch ?? config.actor.netArch.critic
        let criticNormalizeImages = config.critic.normalizeImages ?? config.actor.normalizeImages
        let criticActivation = config.critic.activation ?? config.actor.activation
        let criticExtractorConfig =
            config.critic.featuresExtractor ?? config.actor.featuresExtractor

        let criticExtractor: (any FeaturesExtractor)? =
            config.critic.shareFeaturesExtractor
            ? actor.featuresExtractor
            : try criticExtractorConfig.make(
                observationSpace: observationSpace,
                normalizeImages: criticNormalizeImages
            )

        let critic = try SACCritic(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            normalizeImages: criticNormalizeImages,
            netArch: criticNetArch,
            featuresExtractor: criticExtractor,
            nCritics: config.critic.nCritics,
            shareFeaturesExtractor: config.critic.shareFeaturesExtractor,
            activation: { criticActivation.make() },
            initializationKey: nextInitializationKey()
        )

        let targetExtractor = try criticExtractorConfig.make(
            observationSpace: observationSpace,
            normalizeImages: criticNormalizeImages
        )
        let criticTarget = try SACCritic(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            normalizeImages: criticNormalizeImages,
            netArch: criticNetArch,
            featuresExtractor: targetExtractor,
            nCritics: config.critic.nCritics,
            shareFeaturesExtractor: false,
            activation: { criticActivation.make() },
            initializationKey: nextInitializationKey()
        )

        self.actor = actor
        self.critic = critic
        self.criticTarget = criticTarget

        try syncCriticTargetFromCritic()
    }

    public init(
        actor: SACActor,
        criticConfig: SACCriticConfig = SACCriticConfig()
    ) throws {
        let observationSpace = actor.observationSpace
        let actionSpace = actor.actionSpace

        let criticNetArch = criticConfig.netArch ?? actor.netArch.critic
        let criticNormalizeImages = criticConfig.normalizeImages ?? actor.normalizeImages
        let criticActivation = criticConfig.activation ?? .relu
        let criticExtractorConfig = criticConfig.featuresExtractor ?? .auto

        let criticExtractor: (any FeaturesExtractor)? =
            criticConfig.shareFeaturesExtractor
            ? actor.featuresExtractor
            : try criticExtractorConfig.make(
                observationSpace: observationSpace,
                normalizeImages: criticNormalizeImages
            )

        let critic = try SACCritic(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            normalizeImages: criticNormalizeImages,
            netArch: criticNetArch,
            featuresExtractor: criticExtractor,
            nCritics: criticConfig.nCritics,
            shareFeaturesExtractor: criticConfig.shareFeaturesExtractor,
            activation: { criticActivation.make() }
        )

        let targetExtractor = try criticExtractorConfig.make(
            observationSpace: observationSpace,
            normalizeImages: criticNormalizeImages
        )
        let criticTarget = try SACCritic(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            normalizeImages: criticNormalizeImages,
            netArch: criticNetArch,
            featuresExtractor: targetExtractor,
            nCritics: criticConfig.nCritics,
            shareFeaturesExtractor: false,
            activation: { criticActivation.make() }
        )

        self.actor = actor
        self.critic = critic
        self.criticTarget = criticTarget

        try syncCriticTargetFromCritic()
    }
}
