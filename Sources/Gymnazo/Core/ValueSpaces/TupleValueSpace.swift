public struct TupleValueSpace: ValueBatchingSpace, SpaceValueConvertible {
    public let spaces: [AnyValueSpace]

    public init(_ spaces: [AnyValueSpace]) {
        self.spaces = spaces
    }

    public func containsValue(_ value: [SpaceValue]) -> Bool {
        guard value.count == spaces.count else { return false }
        return zip(spaces, value).allSatisfy { space, element in
            space.containsValue(element)
        }
    }

    public func sample(using generator: inout SeededGenerator) throws -> [SpaceValue] {
        try spaces.map { space in
            try space.sample(using: &generator)
        }
    }

    public func encodeValue(_ value: [SpaceValue]) -> SpaceValue {
        .tuple(value)
    }

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
