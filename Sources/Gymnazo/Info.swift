import MLX

public struct Info: ExpressibleByDictionaryLiteral {
    public var storage: [String: InfoValue]

    public init(_ storage: [String: InfoValue] = [:]) {
        self.storage = storage
    }

    init(valueInfo: ValueInfo) {
        self.storage = valueInfo.mapValues(InfoValue.init)
    }

    public init(dictionaryLiteral elements: (String, InfoValue)...) {
        var s: [String: InfoValue] = [:]
        s.reserveCapacity(elements.count)
        for (k, v) in elements {
            s[k] = v
        }
        self.storage = s
    }

    public subscript(_ key: String) -> InfoValue? {
        get { storage[key] }
        set { storage[key] = newValue }
    }

    public var isEmpty: Bool { storage.isEmpty }
    public var count: Int { storage.count }
}

extension InfoValue {
    fileprivate init(_ value: SpaceValue) {
        switch value {
        case .bool(let value):
            self = .bool(value)
        case .double(let value):
            self = .double(value)
        case .continuous(let elements, let shape):
            self = .tensor(MLXArray(elements).reshaped(shape))
        case .discrete(let value):
            self = .int(value)
        case .multiDiscrete(let values):
            self = .array(values.map(InfoValue.int))
        case .multiBinary(let values):
            self = .array(values.map(InfoValue.bool))
        case .text(let value):
            self = .string(value)
        case .tuple(let values), .sequence(let values):
            self = .array(values.map(InfoValue.init))
        case .dictionary(let values):
            self = .object(values.mapValues(InfoValue.init))
        case .floatTensor(let value):
            self = .tensor(MLXArray(value.elements).reshaped(value.shape))
        case .intTensor(let value):
            self = .tensor(MLXArray(value.elements.map(Int32.init)).reshaped(value.shape))
        case .boolTensor(let value):
            self = .tensor(MLXArray(value.elements).reshaped(value.shape))
        case .graph(let value):
            self = .sendable(value)
        }
    }
}

public enum InfoValue {
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([InfoValue])
    case object([String: InfoValue])
    case sendable(any Sendable)
    case tensor(MLXArray)

    public init(_ value: Bool) { self = .bool(value) }
    public init(_ value: Int) { self = .int(value) }
    public init(_ value: Float) { self = .double(Double(value)) }
    public init(_ value: Double) { self = .double(value) }
    public init(_ value: String) { self = .string(value) }
}

extension InfoValue {
    public var bool: Bool? {
        if case .bool(let v) = self { return v }
        return nil
    }

    public var int: Int? {
        if case .int(let v) = self { return v }
        return nil
    }

    public var double: Double? {
        if case .double(let v) = self { return v }
        return nil
    }

    public var string: String? {
        if case .string(let v) = self { return v }
        return nil
    }

    public var array: [InfoValue]? {
        if case .array(let v) = self { return v }
        return nil
    }

    public var object: [String: InfoValue]? {
        if case .object(let v) = self { return v }
        return nil
    }

    public var sendable: (any Sendable)? {
        if case .sendable(let v) = self { return v }
        return nil
    }

    public var tensor: MLXArray? {
        if case .tensor(let v) = self { return v }
        return nil
    }

    public func cast<T>(_ type: T.Type) -> T? {
        switch self {
        case .bool(let v):
            return v as? T
        case .int(let v):
            return v as? T
        case .double(let v):
            return v as? T
        case .string(let v):
            return v as? T
        case .array(let v):
            return v as? T
        case .object(let v):
            return v as? T
        case .sendable(let v):
            if let result = v as? T {
                return result
            }
            return nil
        case .tensor(let v):
            return v as? T
        }
    }
}

extension InfoValue: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: Bool) {
        self = .bool(value)
    }
}

extension InfoValue: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self = .int(value)
    }
}

extension InfoValue: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self = .double(value)
    }
}

extension InfoValue: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) {
        self = .string(value)
    }
}

extension InfoValue: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: InfoValue...) {
        self = .array(elements)
    }
}

extension InfoValue: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (String, InfoValue)...) {
        var s: [String: InfoValue] = [:]
        s.reserveCapacity(elements.count)
        for (k, v) in elements {
            s[k] = v
        }
        self = .object(s)
    }
}
