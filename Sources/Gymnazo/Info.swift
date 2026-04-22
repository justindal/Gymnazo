import MLX

public struct Info: Sendable, ExpressibleByDictionaryLiteral {
    public var storage: [String: InfoValue]

    public init(_ storage: [String: InfoValue] = [:]) {
        self.storage = storage
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

public struct MLXArrayBox: @unchecked Sendable {
    public let array: MLXArray
    public init(array: MLXArray) { self.array = array }
}

public enum InfoValue: Sendable {
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([InfoValue])
    case object([String: InfoValue])
    case sendable(any Sendable)

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
            if let box = v as? MLXArrayBox, T.self == MLXArray.self {
                return box.array as? T
            }
            return nil
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
