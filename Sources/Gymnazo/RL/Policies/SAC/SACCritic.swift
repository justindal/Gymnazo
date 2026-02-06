//
//  SACCritic.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Critic network for SAC that outputs Q(s, a) values.
///
/// Takes observations and scaled actions (in [-1, 1]) and outputs Q-values
/// from multiple critic networks for clipped double Q-learning.
public final class SACCritic: Module, ContinuousCritic {
    public let observationSpace: any Space
    private let _actionSpace: any Space
    public var actionSpace: any Space { _actionSpace }
    public let normalizeImages: Bool
    public let netArch: [Int]
    public let featuresDim: Int
    public let nCritics: Int
    public let shareFeaturesExtractor: Bool

    public let featuresExtractor: (any FeaturesExtractor)?

    /// Features extractor accessor.
    public var extractor: any FeaturesExtractor {
        guard let ext = featuresExtractor else {
            preconditionFailure("SACCritic requires a features extractor")
        }
        return ext
    }

    @ModuleInfo public var qNetworks: [Sequential]

    public init(
        observationSpace: any Space,
        actionSpace: any Space,
        normalizeImages: Bool = true,
        netArch: [Int] = [256, 256],
        featuresExtractor: (any FeaturesExtractor)? = nil,
        nCritics: Int = 2,
        shareFeaturesExtractor: Bool = false,
        activation: @escaping () -> any UnaryLayer = { ReLU() }
    ) {
        self.observationSpace = observationSpace
        self._actionSpace = actionSpace
        self.normalizeImages = normalizeImages
        self.netArch = netArch
        self.nCritics = nCritics
        self.shareFeaturesExtractor = shareFeaturesExtractor

        let extractor =
            featuresExtractor
            ?? FeaturesExtractorConfig.auto.make(
                observationSpace: observationSpace,
                normalizeImages: normalizeImages
            )
        self.featuresExtractor = extractor
        self.featuresDim = extractor.featuresDim

        let actionDim = getActionDim(actionSpace)
        self.qNetworks = createQNetworks(
            featuresDim: featuresDim,
            actionDim: actionDim,
            netArch: netArch,
            nCritics: nCritics,
            activation: activation
        )

        super.init()
    }

    public func makeFeatureExtractor() -> any FeaturesExtractor {
        FlattenExtractor(featuresDim: observationSpace.shape?.reduce(1, *) ?? 1)
    }
}
