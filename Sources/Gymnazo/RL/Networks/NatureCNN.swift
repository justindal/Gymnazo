//
//  NatureCNN.swift
//  Gymnazo
//
//

import MLX
import MLXNN

@inline(__always)
private func convOut(
    _ input: Int,
    _ kernel: Int,
    _ stride: Int,
    _ padding: Int = 0
) -> Int {
    return (input + 2 * padding - kernel) / stride + 1
}

/// CNN from DQN Nature paper:
/// Mnih, Volodymyr, et al.
/// "Human-level control through deep reinforcement learning."
/// Nature 518.7540 (2015)
/// `NatureCNN` is a convolutional feature extractor intended for image observations.
/// It maps an input image tensor to a fixed-size latent feature vector.
///
/// - Parameters:
///     - observationSpace: The observation space of the environment. Must be a ``Box`` with 3 dimensions `[H, W, C]`.
///     - featuresDim: Number of features extracted (output feature dimension). This corresponds to the number of
///       units in the final linear layer.
///     - normalizedImage: Whether to assume the image is already normalized.
///         - If `true`, only the shape is validated (Box with 3 dimensions). Dtype/bounds checks are skipped.
///         - If `false`, the space is expected to use `uint8` inputs (typically representing pixel values in `[0, 255]`).
public final class NatureCNN: Module, UnaryLayer, FeaturesExtractor {
    public let featuresDim: Int
    private let normalizedImage: Bool
    private let expectedShape: [Int]
    private let nFlatten: Int

    @ModuleInfo private var cnn: Sequential
    @ModuleInfo private var head: Sequential

    public init(
        observationSpace: Box,
        featuresDim: Int = 512,
        normalizedImage: Bool = false
    ) {
        precondition(featuresDim > 0)
        self.featuresDim = featuresDim
        self.normalizedImage = normalizedImage

        guard let shape = observationSpace.shape, shape.count == 3 else {
            preconditionFailure("Shape should be [H, W, C].")
        }

        self.expectedShape = shape

        let h = shape[0]
        let w = shape[1]
        let c = shape[2]
        precondition(h > 0 && w > 0 && c > 0)

        if !normalizedImage {
            precondition(
                observationSpace.dtype == .uint8,
                "Expected type .uint8."
            )
        }

        self.cnn = Sequential {
            Conv2d(
                inputChannels: c,
                outputChannels: 32,
                kernelSize: 8,
                stride: 4,
                padding: 0
            )
            ReLU()
            Conv2d(
                inputChannels: 32,
                outputChannels: 64,
                kernelSize: 4,
                stride: 2,
                padding: 0
            )
            ReLU()
            Conv2d(
                inputChannels: 64,
                outputChannels: 64,
                kernelSize: 3,
                stride: 1,
                padding: 0
            )
            ReLU()
        }

        let h1 = convOut(h, 8, 4)
        let w1 = convOut(w, 8, 4)
        let h2 = convOut(h1, 4, 2)
        let w2 = convOut(w1, 4, 2)
        let h3 = convOut(h2, 3, 1)
        let w3 = convOut(w2, 3, 1)
        precondition(
            h3 > 0 && w3 > 0,
            "Input image too small for NatureCNN convolutions (need at least ~36x36)."
        )

        let nFlatten = 64 * h3 * w3

        self.head = Sequential {
            Linear(nFlatten, featuresDim)
            ReLU()
        }

        self.nFlatten = nFlatten

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        let batched: Bool

        if s == self.expectedShape {
            batched = false
        } else if s.count == 4 && Array(s.dropFirst()) == expectedShape {
            batched = true
        } else {
            preconditionFailure(
                "NatureCNN got shape \(s), expected \(expectedShape) or [N, \(expectedShape.map(String.init).joined(separator: ", "))]."
            )
        }

        var xf = x.asType(.float32)

        if !normalizedImage {
            xf = xf / 255.0
        }

        if !batched {
            xf = xf.reshaped([1] + expectedShape)
        }

        var y = cnn(xf)

        precondition(
            y.shape.count == 4,
            "Conv output should have shape [N, H', W', 64] but got \(y.shape)."
        )
        let b = y.shape[0]

        y = y.reshaped([b, nFlatten])

        let result = head(y)

        if !batched {
            return result.squeezed(axis: 0)
        }
        return result
    }
}
