/// A typed tensor space containing only boolean values.
public struct MultiBinaryValueSpace: ValueBatchingSpace {
    public let shape: [Int]

    /// Creates a boolean tensor space with the supplied shape.
    ///
    /// - Parameter shape: Tensor dimensions.
    /// - Throws: If the shape is invalid.
    public init(shape: [Int]) throws {
        _ = try TensorValue<Bool>.elementCount(for: shape)
        self.shape = shape
    }

    /// Creates a one-dimensional boolean space with n values.
    ///
    /// - Parameter n: Number of boolean elements.
    /// - Throws: If n is invalid.
    public init(n: Int) throws {
        try self.init(shape: [n])
    }

    /// Returns whether a boolean tensor matches the declared shape.
    ///
    /// - Parameter value: The tensor to validate.
    /// - Returns: True when its shape matches.
    public func containsValue(_ value: TensorValue<Bool>) -> Bool {
        value.shape == shape
    }

    /// Samples a boolean tensor.
    ///
    /// - Parameter generator: The random generator used for sampling.
    /// - Returns: A sampled boolean tensor.
    /// - Throws: If tensor construction fails.
    public func sample(using generator: inout SeededGenerator) throws
        -> TensorValue<Bool>
    {
        let count = try TensorValue<Bool>.elementCount(for: shape)
        let values = (0..<count).map { _ in generator.next() & 1 == 1 }
        return try TensorValue(values, shape: shape)
    }
}
