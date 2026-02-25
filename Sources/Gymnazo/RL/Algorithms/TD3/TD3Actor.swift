//
//  TD3Actor.swift
//  Gymnazo
//
//  Created by Justin Daludado on 2026-02-23.
//

import MLX
import MLXNN

public final class TD3Actor: Module, Policy, @unchecked Sendable {
    public let observationSpace: any Space
    public let actionSpace: any Space
    public let netArch: NetArch
    public let featuresExtractor: any FeaturesExtractor
    public let featuresDim: Int
    public let normalizeImages: Bool

    @ModuleInfo private var mu: Sequential

    public init(
        observationSpace: any Space,
        actionSpace: any Space,
        netArch: NetArch = .shared([400, 300]),
        featuresExtractor: (any FeaturesExtractor)? = nil,
        normalizeImages: Bool = true,
        activation: @escaping () -> any UnaryLayer = { ReLU() }
    ) {
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        self.netArch = netArch
        self.normalizeImages = normalizeImages

        let extractor =
            featuresExtractor
            ?? FeaturesExtractorConfig.auto.make(
                observationSpace: observationSpace,
                normalizeImages: normalizeImages
            )
        self.featuresExtractor = extractor
        self.featuresDim = extractor.featuresDim
        let actionDim = getActionDim(actionSpace)

        self.mu = MLPFactory.make(
            inputDim: featuresDim,
            outputDim: actionDim,
            hiddenLayers: netArch.actor,
            activation: activation
        )

        super.init()
    }

    public convenience init(
        observationSpace: any Space,
        actionSpace: any Space,
        config: TD3ActorConfig = TD3ActorConfig()
    ) {
        self.init(
            observationSpace: observationSpace,
            actionSpace: actionSpace,
            netArch: config.netArch,
            featuresExtractor: config.featuresExtractor.make(
                observationSpace: observationSpace,
                normalizeImages: config.normalizeImages
            ),
            normalizeImages: config.normalizeImages,
            activation: { config.activation.make() }
        )
    }

    public var squashOutput: Bool { true }

    public func callAsFunction(_ observation: MLXArray, deterministic _: Bool)
        -> MLXArray
    {
        MLX.tanh(mu(observation))
    }

    public func makeFeatureExtractor() -> any FeaturesExtractor {
        FlattenExtractor(featuresDim: observationSpace.shape?.reduce(1, *) ?? 1)
    }
}
