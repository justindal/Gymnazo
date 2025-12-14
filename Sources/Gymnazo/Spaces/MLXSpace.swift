import MLX

public protocol MLXSpace: Space where T == MLXArray {
    func sampleBatch(key: MLXArray, count: Int) -> MLXArray
}

