import MLX

public protocol TensorSpace: Space {
    func sampleBatch(key: MLXArray, count: Int) -> MLXArray
}
