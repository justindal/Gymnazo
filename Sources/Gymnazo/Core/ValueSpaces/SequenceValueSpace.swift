/// A variable-length sequence of values from one inner space.
public struct SequenceValueSpace<Inner: ValueSpace>: ValueBatchingSpace {
    public let space: Inner
    public let minLength: Int
    public let maxLength: Int

    /// Creates a sequence space with inclusive minimum and maximum lengths.
    ///
    /// - Parameters:
    ///   - space: Space for each sequence element.
    ///   - minLength: Minimum permitted sequence length.
    ///   - maxLength: Maximum permitted sequence length.
    /// - Throws: If length bounds are invalid.
    public init(space: Inner, minLength: Int = 0, maxLength: Int) throws {
        guard minLength >= 0, maxLength >= minLength else {
            throw GymnazoError.invalidConfiguration("Sequence length bounds are invalid.")
        }
        self.space = space
        self.minLength = minLength
        self.maxLength = maxLength
    }

    /// Returns whether length and every element satisfy the space.
    ///
    /// - Parameter value: The sequence to validate.
    /// - Returns: True when length and elements match the space.
    public func containsValue(_ value: [Inner.Value]) -> Bool {
        (minLength...maxLength).contains(value.count) && value.allSatisfy(space.containsValue)
    }

    /// Samples a deterministic length and one value per element.
    ///
    /// - Parameter generator: The random generator used for sampling.
    /// - Returns: A sampled sequence.
    /// - Throws: If the inner space cannot sample.
    public func sample(using generator: inout SeededGenerator) throws -> [Inner.Value] {
        let length = Int.random(in: minLength...maxLength, using: &generator)
        return try (0..<length).map { _ in
            try space.sample(using: &generator)
        }
    }
}

extension SequenceValueSpace: SpaceValueConvertible where Inner: SpaceValueConvertible {
    /// Erases a sequence as SpaceValue.
    ///
    /// - Parameter value: The sequence to encode.
    /// - Returns: The type-erased sequence representation.
    public func encodeValue(_ value: [Inner.Value]) -> SpaceValue {
        .sequence(value.map(space.encodeValue))
    }

    /// Decodes a type-erased sequence when its case matches.
    ///
    /// - Parameter value: The erased value to decode.
    /// - Returns: The typed sequence, or nil for another value case.
    public func decodeValue(_ value: SpaceValue) -> [Inner.Value]? {
        guard case .sequence(let values) = value else { return nil }
        var decoded: [Inner.Value] = []
        decoded.reserveCapacity(values.count)
        for value in values {
            guard let element = space.decodeValue(value) else { return nil }
            decoded.append(element)
        }
        return decoded
    }
}

extension SequenceValueSpace: ValueFlatteningSpace where Inner: ValueFlatteningSpace {
    public var flatDimension: Int? {
        nil
    }

    /// Flattens each sequence element while preserving sequence structure.
    ///
    /// - Parameter value: The sequence to flatten.
    /// - Returns: A flattened sequence representation.
    /// - Throws: If the sequence is outside the space.
    public func flattenValue(_ value: [Inner.Value]) throws -> FlattenedValue {
        guard containsValue(value) else {
            throw GymnazoError.invalidState("Cannot flatten a value outside the sequence space.")
        }
        return .sequence(try value.map(space.flattenValue))
    }

    /// Restores sequence elements from a flattened value.
    ///
    /// - Parameter value: The flattened sequence representation.
    /// - Returns: The restored sequence.
    /// - Throws: If the representation is invalid for the space.
    public func unflattenValue(_ value: FlattenedValue) throws -> [Inner.Value] {
        guard case .sequence(let values) = value else {
            throw GymnazoError.invalidState("The flattened sequence structure is invalid.")
        }
        let result = try values.map(space.unflattenValue)
        guard containsValue(result) else {
            throw GymnazoError.invalidState("The unflattened sequence value is outside the space.")
        }
        return result
    }
}
