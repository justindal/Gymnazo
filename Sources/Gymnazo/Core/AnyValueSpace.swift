public protocol SpaceValueConvertible: ValueSpace {
    func encodeValue(_ value: Value) -> SpaceValue
    func decodeValue(_ value: SpaceValue) -> Value?
}

public struct AnyValueSpace: ValueBatchingSpace {
    private let contains: @Sendable (SpaceValue) -> Bool
    private let sampleValue: @Sendable (inout SeededGenerator) throws -> SpaceValue
    private let flatten: @Sendable (SpaceValue) throws -> FlattenedValue
    private let unflatten: @Sendable (FlattenedValue) throws -> SpaceValue
    public let flatDimension: Int?

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

    public func containsValue(_ value: SpaceValue) -> Bool {
        contains(value)
    }

    public func sample(using generator: inout SeededGenerator) throws -> SpaceValue {
        try sampleValue(&generator)
    }

    public func flattenValue(_ value: SpaceValue) throws -> FlattenedValue {
        try flatten(value)
    }

    public func unflattenValue(_ value: FlattenedValue) throws -> SpaceValue {
        try unflatten(value)
    }
}

extension DiscreteValueSpace: SpaceValueConvertible {
    public func encodeValue(_ value: Int) -> SpaceValue {
        .discrete(value)
    }

    public func decodeValue(_ value: SpaceValue) -> Int? {
        guard case .discrete(let value) = value else { return nil }
        return value
    }
}

extension DiscreteValueSpace: ValueFlatteningSpace {
    public var flatDimension: Int? {
        n
    }

    public func flattenValue(_ value: Int) throws -> FlattenedValue {
        guard containsValue(value) else {
            throw GymnazoError.invalidState("Cannot flatten a value outside the discrete space.")
        }
        var flattened = [Float](repeating: 0, count: n)
        flattened[value - start] = 1
        return .vector(flattened)
    }

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
    public func encodeValue(_ value: TensorValue<Float>) -> SpaceValue {
        .floatTensor(value)
    }

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

    public func flattenValue(_ value: TensorValue<Float>) throws -> FlattenedValue {
        guard containsValue(value) else {
            throw GymnazoError.invalidState("Cannot flatten a value outside the box space.")
        }
        return .vector(value.elements)
    }

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
    public func encodeValue(_ value: TensorValue<Bool>) -> SpaceValue {
        .boolTensor(value)
    }

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

    public func flattenValue(_ value: TensorValue<Bool>) throws -> FlattenedValue {
        guard containsValue(value) else {
            throw GymnazoError.invalidState(
                "Cannot flatten a value outside the multi-binary space.")
        }
        return .vector(value.elements.map { $0 ? 1 : 0 })
    }

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
    public func encodeValue(_ value: TensorValue<Int>) -> SpaceValue {
        .intTensor(value)
    }

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
    public func encodeValue(_ value: String) -> SpaceValue {
        .text(value)
    }

    public func decodeValue(_ value: SpaceValue) -> String? {
        guard case .text(let value) = value else { return nil }
        return value
    }
}
