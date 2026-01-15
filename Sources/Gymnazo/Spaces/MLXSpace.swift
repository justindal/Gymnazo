import MLX

/// A ``Space`` whose samples are `MLXArray` tensors.
public protocol MLXSpace: Space<MLXArray> {
    /// Samples `count` elements and returns a tensor with a leading batch dimension.
    func sampleBatch(key: MLXArray, count: Int) -> MLXArray
}

