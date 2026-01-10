//
//  MLPFactory.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Create a multi layer perceptron (MLP), which is a
/// collection of fully connected layers each followed
/// by an activation function.
///
/// - Parameters:
///     - inputDim: Dimension of the input vector.
///     - outputDim: Dimension of the output vector.
///     - hiddenLayers: The hidden layers of the network.
///     - activation: The activation function to use for each layer.
///     - squashOutput: Whether to squash the output using a Tanh activation function.
///     - withBias: If set to False, the layers will not learn an additive bias.
///     - preLinear: List of nn.Module to add before the linear layers.
///     - postLinear: List of nn.Module to add after the linear layers.

public enum MLPFactory {
    public static func make(
        inputDim: Int,
        outputDim: Int,
        hiddenLayers: [Int],
        activation: () -> any UnaryLayer = { ReLU() },
        withBias: Bool = true,
        squashOutput: Bool = false,
        preLinear: ((Int) -> any UnaryLayer)? = nil,
        postLinear: ((Int) -> any UnaryLayer)? = nil
    ) -> Sequential {
        var layers: [any UnaryLayer] = []
        var currentDim = inputDim

        for hidden in hiddenLayers {
            if let pre = preLinear {
                layers.append(pre(currentDim))
            }

            layers.append(Linear(currentDim, hidden, bias: withBias))

            if let post = postLinear {
                layers.append(post(hidden))
            }

            layers.append(activation())
            currentDim = hidden
        }

        if outputDim > 0 {
            if let pre = preLinear {
                layers.append(pre(currentDim))
            }

            layers.append(Linear(currentDim, outputDim, bias: withBias))

            if squashOutput {

            }
        }
        return Sequential(layers: layers)
    }
}
