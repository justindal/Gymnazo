/// A deterministic string-keyed product of typed value spaces.
public struct DictionaryValueSpace: ValueBatchingSpace, SpaceValueConvertible {
    public let spaces: [String: AnyValueSpace]

    /// Creates a dictionary space from keyed erased spaces.
    ///
    /// - Parameter spaces: Spaces keyed by their observation names.
    public init(_ spaces: [String: AnyValueSpace]) {
        self.spaces = spaces
    }

    /// Returns whether all required keys and values belong to their spaces.
    ///
    /// - Parameter value: The dictionary value to validate.
    /// - Returns: True when keys and values match the space.
    public func containsValue(_ value: [String: SpaceValue]) -> Bool {
        guard value.count == spaces.count, value.keys.allSatisfy({ spaces[$0] != nil }) else {
            return false
        }
        return spaces.allSatisfy { key, space in
            guard let element = value[key] else { return false }
            return space.containsValue(element)
        }
    }

    /// Samples keys in stable sorted order.
    ///
    /// - Parameter generator: The random generator used for sampling.
    /// - Returns: A sampled dictionary value.
    /// - Throws: If an inner space cannot sample.
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

    /// Erases a dictionary value as SpaceValue.
    ///
    /// - Parameter value: The dictionary value to encode.
    /// - Returns: The type-erased dictionary representation.
    public func encodeValue(_ value: [String: SpaceValue]) -> SpaceValue {
        .dictionary(value)
    }

    /// Decodes a dictionary SpaceValue when its case matches.
    ///
    /// - Parameter value: The erased value to decode.
    /// - Returns: The dictionary, or nil for another value case.
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

    /// Flattens keyed values in stable key order.
    ///
    /// - Parameter value: The dictionary value to flatten.
    /// - Returns: A flattened dictionary or vector representation.
    /// - Throws: If the value is outside the space.
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

    /// Restores keyed values from a flattened dictionary.
    ///
    /// - Parameter value: The flattened dictionary representation.
    /// - Returns: The restored dictionary value.
    /// - Throws: If the representation is invalid for the space.
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
