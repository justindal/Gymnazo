public struct DictionaryValueSpace: ValueBatchingSpace, SpaceValueConvertible {
    public let spaces: [String: AnyValueSpace]

    public init(_ spaces: [String: AnyValueSpace]) {
        self.spaces = spaces
    }

    public func containsValue(_ value: [String: SpaceValue]) -> Bool {
        guard value.count == spaces.count, value.keys.allSatisfy({ spaces[$0] != nil }) else {
            return false
        }
        return spaces.allSatisfy { key, space in
            guard let element = value[key] else { return false }
            return space.containsValue(element)
        }
    }

    public func sample(using generator: inout SeededGenerator) throws -> [String: SpaceValue] {
        var result: [String: SpaceValue] = [:]
        result.reserveCapacity(spaces.count)
        for key in spaces.keys.sorted() {
            guard let space = spaces[key] else {
                throw GymnazoError.invalidState("A dictionary space lost a registered key.")
            }
            result[key] = try space.sample(using: &generator)
        }
        return result
    }

    public func encodeValue(_ value: [String: SpaceValue]) -> SpaceValue {
        .dictionary(value)
    }

    public func decodeValue(_ value: SpaceValue) -> [String: SpaceValue]? {
        guard case .dictionary(let value) = value else { return nil }
        return value
    }
}

extension DictionaryValueSpace: ValueFlatteningSpace {
    public var flatDimension: Int? {
        let dimensions = spaces.values.compactMap(\.flatDimension)
        guard dimensions.count == spaces.count else { return nil }
        return dimensions.reduce(0, +)
    }

    public func flattenValue(_ value: [String: SpaceValue]) throws -> FlattenedValue {
        guard containsValue(value) else {
            throw GymnazoError.invalidState("Cannot flatten a value outside the dictionary space.")
        }
        var flattened: [String: FlattenedValue] = [:]
        for key in spaces.keys.sorted() {
            guard let space = spaces[key], let element = value[key] else {
                throw GymnazoError.invalidState("A dictionary space lost a registered key.")
            }
            flattened[key] = try space.flattenValue(element)
        }
        if flatDimension != nil {
            var vector: [Float] = []
            vector.reserveCapacity(flatDimension ?? 0)
            for key in spaces.keys.sorted() {
                guard case .vector(let values) = flattened[key] else {
                    throw GymnazoError.invalidState(
                        "A fixed dictionary subspace did not flatten to a vector."
                    )
                }
                vector.append(contentsOf: values)
            }
            return .vector(vector)
        }
        return .dictionary(flattened)
    }

    public func unflattenValue(_ value: FlattenedValue) throws -> [String: SpaceValue] {
        var result: [String: SpaceValue] = [:]
        if let flatDimension {
            guard case .vector(let flattened) = value, flattened.count == flatDimension else {
                throw GymnazoError.invalidState(
                    "The flattened dictionary value has an invalid shape.")
            }
            var offset = 0
            for key in spaces.keys.sorted() {
                guard let space = spaces[key], let dimension = space.flatDimension else {
                    throw GymnazoError.invalidState(
                        "A dictionary subspace lost its flat dimension.")
                }
                result[key] = try space.unflattenValue(
                    .vector(Array(flattened[offset..<(offset + dimension)]))
                )
                offset += dimension
            }
        } else {
            guard case .dictionary(let flattened) = value, flattened.count == spaces.count else {
                throw GymnazoError.invalidState("The flattened dictionary structure is invalid.")
            }
            for key in spaces.keys.sorted() {
                guard let space = spaces[key], let element = flattened[key] else {
                    throw GymnazoError.invalidState("The flattened dictionary is missing a key.")
                }
                result[key] = try space.unflattenValue(element)
            }
        }
        guard containsValue(result) else {
            throw GymnazoError.invalidState(
                "The unflattened dictionary value is outside the space.")
        }
        return result
    }
}
