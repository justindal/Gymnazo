/// A value space that can encode its associated value as ``SpaceValue``.
public protocol SpaceValueConvertible: ValueSpace {
    /// Encodes a typed value as a type-erased value.
    ///
    /// - Parameter value: The typed value to encode.
    /// - Returns: The corresponding SpaceValue.
    func encodeValue(_ value: Value) -> SpaceValue
    /// Decodes a type-erased value as the associated typed value.
    ///
    /// - Parameter value: The erased value to decode.
    /// - Returns: The typed value, or nil when the case does not match.
    func decodeValue(_ value: SpaceValue) -> Value?
}

/// Type-erases a typed space while preserving validation and batching.
public struct AnyValueSpace: ValueBatchingSpace {
    private let contains: @Sendable (SpaceValue) -> Bool
    private let sampleValue: @Sendable (inout SeededGenerator) throws -> SpaceValue
    private let flatten: @Sendable (SpaceValue) throws -> FlattenedValue
    private let unflatten: @Sendable (FlattenedValue) throws -> SpaceValue
    public let flatDimension: Int?

    /// Creates a type-erased space from a value-convertible space.
    ///
    /// - Parameter base: The typed space to erase.
    public init<Base: SpaceValueConvertible>(_ base: Base) {
        self.contains = { value in
            guard let decoded = base.decodeValue(value) else { return false }
            return base.containsValue(decoded)
        }
        self.sampleValue = { generator in
            base.encodeValue(try base.sample(using: &generator))
        }
        self.flatten = { _ in
            throw GymnazoError.unsupportedSpaceOperation(
                operation: "flatten",
                spaceType: String(describing: Base.self)
            )
        }
        self.unflatten = { _ in
            throw GymnazoError.unsupportedSpaceOperation(
                operation: "unflatten",
                spaceType: String(describing: Base.self)
            )
        }
        self.flatDimension = nil
    }

    /// Creates a type-erased space that also supports reversible flattening.
    ///
    /// - Parameter base: The flattening space to erase.
    public init<Base: ValueFlatteningSpace>(_ base: Base) {
        self.contains = { value in
            guard let decoded = base.decodeValue(value) else { return false }
            return base.containsValue(decoded)
        }
        self.sampleValue = { generator in
            base.encodeValue(try base.sample(using: &generator))
        }
        self.flatten = { value in
            guard let decoded = base.decodeValue(value) else {
                throw GymnazoError.invalidState("Cannot flatten a value outside the space.")
            }
            return try base.flattenValue(decoded)
        }
        self.unflatten = { value in
            base.encodeValue(try base.unflattenValue(value))
        }
        self.flatDimension = base.flatDimension
    }

    /// Returns whether the erased value belongs to the space.
    ///
    /// - Parameter value: The erased value to validate.
    /// - Returns: `true` when the value belongs to the space.
    public func containsValue(_ value: SpaceValue) -> Bool {
        contains(value)
    }

    /// Samples a value and erases it as ``SpaceValue``.
    ///
    /// - Parameter generator: The random generator used for sampling.
    /// - Returns: A sampled erased value.
    /// - Throws: If the underlying space cannot sample a value.
    public func sample(using generator: inout SeededGenerator) throws -> SpaceValue {
        try sampleValue(&generator)
    }

    /// Flattens an erased value when the underlying space supports it.
    ///
    /// - Parameter value: The erased value to flatten.
    /// - Returns: A reversible flattened representation.
    /// - Throws: If the value is invalid or flattening is unsupported.
    public func flattenValue(_ value: SpaceValue) throws -> FlattenedValue {
        try flatten(value)
    }

    /// Restores an erased value from its flattened representation.
    ///
    /// - Parameter value: The flattened value to restore.
    /// - Returns: The restored erased value.
    /// - Throws: If the flattened representation is invalid.
    public func unflattenValue(_ value: FlattenedValue) throws -> SpaceValue {
        try unflatten(value)
    }
}

extension DiscreteValueSpace: SpaceValueConvertible {
    /// Encodes an integer as a discrete SpaceValue.
    ///
    /// - Parameter value: The integer to encode.
    /// - Returns: A discrete type-erased value.
    public func encodeValue(_ value: Int) -> SpaceValue {
        .discrete(value)
    }

    /// Decodes a discrete SpaceValue.
    ///
    /// - Parameter value: The erased value to decode.
    /// - Returns: The integer, or nil for another value case.
    public func decodeValue(_ value: SpaceValue) -> Int? {
        guard case .discrete(let value) = value else { return nil }
        return value
    }
}

extension DiscreteValueSpace: ValueFlatteningSpace {
    public var flatDimension: Int? {
        n
    }

    /// Flattens a discrete value as one-hot elements.
    ///
    /// - Parameter value: The discrete value to flatten.
    /// - Returns: A one-hot vector.
    /// - Throws: If the value is outside the space.
    public func flattenValue(_ value: Int) throws -> FlattenedValue {
        guard containsValue(value) else {
            throw GymnazoError.invalidState("Cannot flatten a value outside the discrete space.")
        }
        var flattened = [Float](repeating: 0, count: n)
        flattened[value - start] = 1
        return .vector(flattened)
    }

    /// Restores a discrete value from a one-hot vector.
    ///
    /// - Parameter value: The flattened one-hot vector.
    /// - Returns: The selected discrete value.
    /// - Throws: If the vector shape is invalid.
    public func unflattenValue(_ value: FlattenedValue) throws -> Int {
        guard case .vector(let values) = value, values.count == n,
            let index = values.indices.max(by: { values[$0] < values[$1] })
        else {
            throw GymnazoError.invalidState("The flattened discrete value has an invalid shape.")
        }
        return start + index
    }
}

extension BoxValueSpace: SpaceValueConvertible {
    /// Encodes a float tensor as a float-tensor SpaceValue.
    ///
    /// - Parameter value: The tensor to encode.
    /// - Returns: A float-tensor type-erased value.
    public func encodeValue(_ value: TensorValue<Float>) -> SpaceValue {
        .floatTensor(value)
    }

    /// Decodes a float tensor SpaceValue.
    ///
    /// - Parameter value: The erased value to decode.
    /// - Returns: The float tensor, or nil for another value case.
    public func decodeValue(_ value: SpaceValue) -> TensorValue<Float>? {
        switch value {
        case .floatTensor(let value):
            return value
        case .continuous(let elements, let shape):
            return try? TensorValue(elements, shape: shape)
        default:
            return nil
        }
    }
}

extension BoxValueSpace: ValueFlatteningSpace {
    public var flatDimension: Int? {
        low.count
    }

    /// Flattens a bounded float tensor.
    ///
    /// - Parameter value: The tensor to flatten.
    /// - Returns: A vector containing tensor elements.
    /// - Throws: If the value is outside the box space.
    public func flattenValue(_ value: TensorValue<Float>) throws -> FlattenedValue {
        guard containsValue(value) else {
            throw GymnazoError.invalidState("Cannot flatten a value outside the box space.")
        }
        return .vector(value.elements)
    }

    /// Restores a bounded float tensor from a vector.
    ///
    /// - Parameter value: The flattened vector.
    /// - Returns: The restored tensor.
    /// - Throws: If shape or bounds are invalid.
    public func unflattenValue(_ value: FlattenedValue) throws -> TensorValue<Float> {
        guard case .vector(let values) = value else {
            throw GymnazoError.invalidState("The flattened box value must be a vector.")
        }
        let result = try TensorValue(values, shape: shape)
        guard containsValue(result) else {
            throw GymnazoError.invalidState("The flattened box value is outside the space.")
        }
        return result
    }
}

extension MultiBinaryValueSpace: SpaceValueConvertible {
    /// Encodes a boolean tensor as a bool-tensor SpaceValue.
    ///
    /// - Parameter value: The tensor to encode.
    /// - Returns: A bool-tensor type-erased value.
    public func encodeValue(_ value: TensorValue<Bool>) -> SpaceValue {
        .boolTensor(value)
    }

    /// Decodes a bool-tensor SpaceValue.
    ///
    /// - Parameter value: The erased value to decode.
    /// - Returns: The boolean tensor, or nil for another value case.
    public func decodeValue(_ value: SpaceValue) -> TensorValue<Bool>? {
        switch value {
        case .boolTensor(let value):
            return value
        case .multiBinary(let elements):
            return try? TensorValue(elements, shape: shape)
        default:
            return nil
        }
    }
}

extension MultiBinaryValueSpace: ValueFlatteningSpace {
    public var flatDimension: Int? {
        try? TensorValue<Bool>.elementCount(for: shape)
    }

    /// Flattens a boolean tensor as zero and one elements.
    ///
    /// - Parameter value: The tensor to flatten.
    /// - Returns: A numeric vector representation.
    /// - Throws: If the value is outside the space.
    public func flattenValue(_ value: TensorValue<Bool>) throws -> FlattenedValue {
        guard containsValue(value) else {
            throw GymnazoError.invalidState(
                "Cannot flatten a value outside the multi-binary space.")
        }
        return .vector(value.elements.map { $0 ? 1 : 0 })
    }

    /// Restores a boolean tensor from a numeric vector.
    ///
    /// - Parameter value: The flattened vector.
    /// - Returns: The restored boolean tensor.
    /// - Throws: If shape is invalid.
    public func unflattenValue(_ value: FlattenedValue) throws -> TensorValue<Bool> {
        guard case .vector(let values) = value else {
            throw GymnazoError.invalidState("The flattened multi-binary value must be a vector.")
        }
        let result = try TensorValue(values.map { $0 != 0 }, shape: shape)
        guard containsValue(result) else {
            throw GymnazoError.invalidState(
                "The flattened multi-binary value is outside the space.")
        }
        return result
    }
}

extension MultiDiscreteValueSpace: SpaceValueConvertible {
    /// Encodes an integer tensor as an int-tensor SpaceValue.
    ///
    /// - Parameter value: The tensor to encode.
    /// - Returns: An int-tensor type-erased value.
    public func encodeValue(_ value: TensorValue<Int>) -> SpaceValue {
        .intTensor(value)
    }

    /// Decodes an int-tensor SpaceValue.
    ///
    /// - Parameter value: The erased value to decode.
    /// - Returns: The integer tensor, or nil for another value case.
    public func decodeValue(_ value: SpaceValue) -> TensorValue<Int>? {
        switch value {
        case .intTensor(let value):
            return value
        case .multiDiscrete(let elements):
            return try? TensorValue(elements, shape: shape)
        default:
            return nil
        }
    }
}

extension MultiDiscreteValueSpace: ValueFlatteningSpace {
    public var flatDimension: Int? {
        nvec.elements.reduce(0, +)
    }

    /// Flattens each integer component as a one-hot segment.
    ///
    /// - Parameter value: The tensor to flatten.
    /// - Returns: Concatenated one-hot segments.
    /// - Throws: If the value is outside the space.
    public func flattenValue(_ value: TensorValue<Int>) throws -> FlattenedValue {
        guard containsValue(value) else {
            throw GymnazoError.invalidState(
                "Cannot flatten a value outside the multi-discrete space.")
        }
        var flattened: [Float] = []
        flattened.reserveCapacity(flatDimension ?? 0)
        for ((element, offset), count) in zip(
            zip(value.elements, start.elements),
            nvec.elements
        ) {
            flattened.append(contentsOf: [Float](repeating: 0, count: count))
            flattened[flattened.count - count + element - offset] = 1
        }
        return .vector(flattened)
    }

    /// Restores integer components from one-hot segments.
    ///
    /// - Parameter value: The flattened segment vector.
    /// - Returns: The restored integer tensor.
    /// - Throws: If the vector shape is invalid.
    public func unflattenValue(_ value: FlattenedValue) throws -> TensorValue<Int> {
        guard case .vector(let values) = value, values.count == flatDimension else {
            throw GymnazoError.invalidState(
                "The flattened multi-discrete value has an invalid shape.")
        }
        var offset = 0
        var result: [Int] = []
        result.reserveCapacity(nvec.count)
        for (start, count) in zip(start.elements, nvec.elements) {
            let slice = values[offset..<(offset + count)]
            guard let index = slice.indices.max(by: { values[$0] < values[$1] }) else {
                throw GymnazoError.invalidState("A multi-discrete segment is empty.")
            }
            result.append(start + index - offset)
            offset += count
        }
        return try TensorValue(result, shape: shape)
    }
}

extension TextValueSpace: SpaceValueConvertible {
    /// Encodes a string as a text SpaceValue.
    ///
    /// - Parameter value: The string to encode.
    /// - Returns: A text type-erased value.
    public func encodeValue(_ value: String) -> SpaceValue {
        .text(value)
    }

    /// Decodes a text SpaceValue.
    ///
    /// - Parameter value: The erased value to decode.
    /// - Returns: The string, or nil for another value case.
    public func decodeValue(_ value: SpaceValue) -> String? {
        guard case .text(let value) = value else { return nil }
        return value
    }
}
