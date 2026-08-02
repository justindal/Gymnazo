import MLX

/// Converts typed tensor values to and from MLX at policy boundaries.
public enum MLXValueCodec {
    /// Encodes a float tensor without changing its shape.
    ///
    /// - Parameter value: The typed tensor to encode.
    /// - Returns: An MLX array with the same shape and elements.
    public static func encode(_ value: TensorValue<Float>) -> MLXArray {
        MLXArray(value.elements, value.shape)
    }

    /// Encodes an integer tensor without changing its shape.
    ///
    /// - Parameter value: The typed tensor to encode.
    /// - Returns: An MLX array with the same shape and elements.
    public static func encode(_ value: TensorValue<Int>) -> MLXArray {
        MLXArray(value.elements, value.shape)
    }

    /// Encodes equally shaped float tensors with a leading batch dimension.
    ///
    /// - Parameter values: Equally shaped tensors in batch order.
    /// - Returns: An MLX array whose leading dimension is the batch count.
    /// - Throws: If the batch is empty or tensor shapes differ.
    public static func encodeBatch(_ values: [TensorValue<Float>]) throws -> MLXArray {
        guard let first = values.first else {
            throw GymnazoError.invalidConfiguration(
                "Cannot encode an empty tensor batch without a shape.")
        }
        guard values.allSatisfy({ $0.shape == first.shape }) else {
            throw GymnazoError.invalidConfiguration(
                "All tensors in a batch must have equal shapes.")
        }
        return MLXArray(values.flatMap(\.elements), [values.count] + first.shape)
    }

    /// Encodes equally shaped integer tensors with a leading batch dimension.
    ///
    /// - Parameter values: Equally shaped tensors in batch order.
    /// - Returns: An MLX array whose leading dimension is the batch count.
    /// - Throws: If the batch is empty or tensor shapes differ.
    public static func encodeBatch(_ values: [TensorValue<Int>]) throws -> MLXArray {
        guard let first = values.first else {
            throw GymnazoError.invalidConfiguration(
                "Cannot encode an empty tensor batch without a shape.")
        }
        guard values.allSatisfy({ $0.shape == first.shape }) else {
            throw GymnazoError.invalidConfiguration(
                "All tensors in a batch must have equal shapes.")
        }
        return MLXArray(values.flatMap(\.elements), [values.count] + first.shape)
    }

    /// Materializes and decodes an MLX array as a typed float tensor.
    ///
    /// - Parameter array: The MLX array to materialize.
    /// - Returns: A typed tensor with the array's shape and values.
    /// - Throws: If the array cannot be represented as a typed tensor.
    public static func decodeFloat(_ array: MLXArray) throws -> TensorValue<Float> {
        eval(array)
        return try TensorValue(array.asArray(Float.self), shape: array.shape)
    }

    /// Materializes and decodes an MLX array as a typed integer tensor.
    ///
    /// - Parameter array: The MLX array to materialize.
    /// - Returns: A typed tensor with the array's shape and values.
    /// - Throws: If the array cannot be represented as a typed tensor.
    public static func decodeInt(_ array: MLXArray) throws -> TensorValue<Int> {
        eval(array)
        return try TensorValue(array.asArray(Int.self), shape: array.shape)
    }
}
