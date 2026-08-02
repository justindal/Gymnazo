/// An ordered product of heterogeneous typed value spaces.
public struct TupleValueSpace: ValueBatchingSpace, SpaceValueConvertible {
    public let spaces: [AnyValueSpace]

    /// Creates a tuple space from ordered erased spaces.
    ///
    /// - Parameter spaces: Element spaces in tuple order.
    public init(_ spaces: [AnyValueSpace]) {
        self.spaces = spaces
    }

    /// Returns whether each tuple element belongs to its corresponding space.
    ///
    /// - Parameter value: The tuple value to validate.
    /// - Returns: True when every element matches its space.
    public func containsValue(_ value: [SpaceValue]) -> Bool {
        guard value.count == spaces.count else { return false }
        return zip(spaces, value).allSatisfy { space, element in
            space.containsValue(element)
        }
    }

    /// Samples each tuple element in order.
    ///
    /// - Parameter generator: The random generator used for sampling.
    /// - Returns: A sampled tuple value.
    /// - Throws: If an element space cannot sample.
    public func sample(using generator: inout SeededGenerator) throws -> [SpaceValue] {
        try spaces.map { space in
            try space.sample(using: &generator)
        }
    }

    /// Erases a tuple value as SpaceValue.
    ///
    /// - Parameter value: The tuple to encode.
    /// - Returns: The type-erased tuple representation.
    public func encodeValue(_ value: [SpaceValue]) -> SpaceValue {
        .tuple(value)
    }

    /// Decodes a tuple SpaceValue when its case matches.
    ///
    /// - Parameter value: The erased value to decode.
    /// - Returns: The tuple, or nil for another value case.
    public func decodeValue(_ value: SpaceValue) -> [SpaceValue]? {
        guard case .tuple(let value) = value else { return nil }
        return value
    }
}

extension TupleValueSpace: ValueFlatteningSpace {
    public var flatDimension: Int? {
        let dimensions = spaces.compactMap(\.flatDimension)
        guard dimensions.count == spaces.count else { return nil }
        return dimensions.reduce(0, +)
    }

    /// Flattens tuple elements in order.
    ///
    /// - Parameter value: The tuple to flatten.
    /// - Returns: A flattened tuple or vector representation.
    /// - Throws: If the value is outside the space.
    public func flattenValue(_ value: [SpaceValue]) throws -> FlattenedValue {
        guard containsValue(value) else {
            throw GymnazoError.invalidState("Cannot flatten a value outside the tuple space.")
        }
        let flattened = try zip(spaces, value).map { space, value in
            try space.flattenValue(value)
        }
        if flatDimension != nil {
            return .vector(try flattened.flatMap(Self.vector))
        }
        return .tuple(flattened)
    }

    /// Restores tuple elements from a flattened tuple.
    ///
    /// - Parameter value: The flattened tuple representation.
    /// - Returns: The restored tuple value.
    /// - Throws: If the representation is invalid for the space.
    public func unflattenValue(_ value: FlattenedValue) throws -> [SpaceValue] {
        let values: [SpaceValue]
        if let flatDimension {
            guard case .vector(let flattened) = value, flattened.count == flatDimension else {
                throw GymnazoError.invalidState("The flattened tuple value has an invalid shape.")
            }
            var offset = 0
            values = try spaces.map { space in
                guard let dimension = space.flatDimension else {
                    throw GymnazoError.invalidState("A tuple subspace lost its flat dimension.")
                }
                defer { offset += dimension }
                return try space.unflattenValue(
                    .vector(Array(flattened[offset..<(offset + dimension)]))
                )
            }
        } else {
            guard case .tuple(let flattened) = value, flattened.count == spaces.count else {
                throw GymnazoError.invalidState("The flattened tuple structure is invalid.")
            }
            values = try zip(spaces, flattened).map { space, value in
                try space.unflattenValue(value)
            }
        }
        guard containsValue(values) else {
            throw GymnazoError.invalidState("The unflattened tuple value is outside the space.")
        }
        return values
    }

    private static func vector(_ value: FlattenedValue) throws -> [Float] {
        guard case .vector(let values) = value else {
            throw GymnazoError.invalidState("A fixed tuple subspace did not flatten to a vector.")
        }
        return values
    }
}
