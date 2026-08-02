import MLX

public enum MLXValueCodec {
    public static func encode(_ value: TensorValue<Float>) -> MLXArray {
        MLXArray(value.elements, value.shape)
    }

    public static func encode(_ value: TensorValue<Int>) -> MLXArray {
        MLXArray(value.elements, value.shape)
    }

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

    public static func decodeFloat(_ array: MLXArray) throws -> TensorValue<Float> {
        eval(array)
        return try TensorValue(array.asArray(Float.self), shape: array.shape)
    }

    public static func decodeInt(_ array: MLXArray) throws -> TensorValue<Int> {
        eval(array)
        return try TensorValue(array.asArray(Int.self), shape: array.shape)
    }
}
