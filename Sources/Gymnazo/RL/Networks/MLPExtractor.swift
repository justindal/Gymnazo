//
//  MLPExtractor.swift
//  Gymnazo
//
//

import MLX
import MLXNN

/// Constructs an MLP that receives the output from a previous feature extractor or
/// the observations and outputs latent representations for the policy network `pi` or
/// the value network `vf`.
///
/// - Parameters:
///   - featureDim: Dimension of the input feature vector.
///   - netArch: Architecture specification for the policy and value networks.
///   - activation: Activation function applied after each hidden linear layer.
///   - withBias: Whether the linear layers include a bias term.
public final class MLPExtractor: Module {
    public let latentDimPi: Int
    public let latentDimVf: Int

    @ModuleInfo private var policyNet: Sequential
    @ModuleInfo private var valueNet: Sequential

    public init(
        featureDim: Int,
        netArch: NetArch,
        activation: () -> any UnaryLayer = { ReLU() },
        withBias: Bool = true
    ) {
        precondition(featureDim > 0)

        let piDims = netArch.actor
        let vfDims = netArch.critic

        // Build policy network
        var piLayers: [any UnaryLayer] = []
        var lastPi = featureDim
        for dim in piDims {
            piLayers.append(Linear(lastPi, dim, bias: withBias))
            piLayers.append(activation())
            lastPi = dim
        }

        // Build value network
        var vfLayers: [any UnaryLayer] = []
        var lastVf = featureDim
        for dim in vfDims {
            vfLayers.append(Linear(lastVf, dim, bias: withBias))
            vfLayers.append(activation())
            lastVf = dim
        }

        let policyNetLocal: Sequential =
            piLayers.isEmpty ? Sequential(layers: [Identity()]) : Sequential(layers: piLayers)
        let valueNetLocal: Sequential =
            vfLayers.isEmpty ? Sequential(layers: [Identity()]) : Sequential(layers: vfLayers)

        self.latentDimPi = lastPi
        self.latentDimVf = lastVf
        self.policyNet = policyNetLocal
        self.valueNet = valueNetLocal

        super.init()
    }

    public func callAsFunction(_ features: MLXArray) -> (MLXArray, MLXArray) {
        (policyNet(features), valueNet(features))
    }

    public func forwardActor(_ features: MLXArray) -> MLXArray {
        policyNet(features)
    }

    public func forwardCritic(_ features: MLXArray) -> MLXArray {
        valueNet(features)
    }
}
