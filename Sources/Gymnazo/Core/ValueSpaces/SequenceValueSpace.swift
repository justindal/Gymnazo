public struct SequenceValueSpace<Inner: ValueSpace>: ValueBatchingSpace {
    public let space: Inner
    public let minLength: Int
    public let maxLength: Int

    public init(space: Inner, minLength: Int = 0, maxLength: Int) throws {
        guard minLength >= 0, maxLength >= minLength else {
            throw GymnazoError.invalidConfiguration("Sequence length bounds are invalid.")
        }
        self.space = space
        self.minLength = minLength
        self.maxLength = maxLength
    }

    public func containsValue(_ value: [Inner.Value]) -> Bool {
        (minLength...maxLength).contains(value.count) && value.allSatisfy(space.containsValue)
    }

    public func sample(using generator: inout SeededGenerator) throws -> [Inner.Value] {
        let length = Int.random(in: minLength...maxLength, using: &generator)
        return try (0..<length).map { _ in
            try space.sample(using: &generator)
        }
    }
}

extension SequenceValueSpace: SpaceValueConvertible where Inner: SpaceValueConvertible {
    public func encodeValue(_ value: [Inner.Value]) -> SpaceValue {
        .sequence(value.map(space.encodeValue))
    }

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

    public func flattenValue(_ value: [Inner.Value]) throws -> FlattenedValue {
        guard containsValue(value) else {
            throw GymnazoError.invalidState("Cannot flatten a value outside the sequence space.")
        }
        return .sequence(try value.map(space.flattenValue))
    }

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
