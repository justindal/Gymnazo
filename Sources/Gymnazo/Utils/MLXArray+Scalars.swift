import MLX

extension MLXArray {
    /// Returns a scalar value and asserts the array is truly scalar (shape []).
    func scalarValue<T: HasDType>(_ type: T.Type = T.self) -> T {
        precondition(
            ndim == 0,
            "MLXArray.scalarValue requires a scalar array, got shape \(shape)"
        )
        return item(type)
    }

    /// Returns the only element in the array.
    ///
    /// Accepts either a scalar (`[]`) or shape `[1]`, and rejects anything larger.
    func singletonValue<T: HasDType>(_ type: T.Type = T.self) -> T {
        if ndim == 0 {
            return item(type)
        }
        let values = reshaped([-1]).asArray(type)
        precondition(
            values.count == 1,
            "MLXArray.singletonValue requires exactly one element, got \(values.count) for shape \(shape)"
        )
        return values[0]
    }
}
