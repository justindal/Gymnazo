/// A typed action or observation space backed by Swift values.
///
/// Value spaces keep sampling and containment checks independent from MLX. The
/// associated value is the exact Swift type accepted by the environment.
public protocol ValueSpace<Value>: Sendable {
    associatedtype Value: Sendable

    /// Returns whether a value belongs to the space.
    ///
    /// - Parameter value: The value to validate.
    /// - Returns: `true` when the value belongs to the space.
    func containsValue(_ value: Value) -> Bool
    /// Samples a value using the supplied deterministic generator.
    ///
    /// - Parameter generator: The random generator used for sampling.
    /// - Returns: A sampled value in the space.
    /// - Throws: If sampling is unsupported or configuration is invalid.
    func sample(using generator: inout SeededGenerator) throws -> Value
}

extension ValueSpace {
    /// Samples a value when a concrete space does not implement sampling.
    ///
    /// - Parameter generator: The random generator reserved for sampling.
    /// - Throws: Always, because the space does not provide sampling.
    public func sample(using generator: inout SeededGenerator) throws -> Value {
        throw GymnazoError.unsupportedSpaceOperation(
            operation: "sample",
            spaceType: String(describing: Self.self)
        )
    }
}

/// A batch of values with one value per environment slot.
public struct ValueBatch<Value: Sendable>: Sendable {
    public let values: [Value]

    /// Creates a batch from ordered values.
    ///
    /// - Parameter values: Values in environment order.
    public init(_ values: [Value]) {
        self.values = values
    }

    /// The number of values in the batch.
    public var count: Int {
        values.count
    }
}

/// A value space that can batch and unbatch environment values.
public protocol ValueBatchingSpace: ValueSpace {
    /// Validates and packs values in environment order.
    ///
    /// - Parameter values: Values to validate and pack.
    /// - Returns: A batch preserving the supplied order.
    /// - Throws: If any value is outside the space.
    func batch(_ values: [Value]) throws -> ValueBatch<Value>
    /// Validates and returns the ordered values from a batch.
    ///
    /// - Parameter batch: The batch to validate and unpack.
    /// - Returns: Values in environment order.
    /// - Throws: If any value is outside the space.
    func unbatch(_ batch: ValueBatch<Value>) throws -> [Value]
}

extension ValueBatchingSpace {
    /// Validates and packs values in environment order.
    ///
    /// - Parameter values: Values to validate and pack.
    /// - Returns: A batch preserving the supplied order.
    /// - Throws: If any value is outside the space.
    public func batch(_ values: [Value]) throws -> ValueBatch<Value> {
        guard values.allSatisfy(containsValue) else {
            throw GymnazoError.invalidState("Cannot batch a value outside the space.")
        }
        return ValueBatch(values)
    }

    /// Validates and returns the ordered values from a batch.
    ///
    /// - Parameter batch: The batch to validate and unpack.
    /// - Returns: Values in environment order.
    /// - Throws: If any value is outside the space.
    public func unbatch(_ batch: ValueBatch<Value>) throws -> [Value] {
        guard batch.values.allSatisfy(containsValue) else {
            throw GymnazoError.invalidState("Cannot unbatch a value outside the space.")
        }
        return batch.values
    }
}
