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

public extension InfoValue {
    var bool: Bool? {
        if case let .bool(v) = self { return v }
        return nil
    }

    var int: Int? {
        if case let .int(v) = self { return v }
        return nil
    }

    var double: Double? {
        if case let .double(v) = self { return v }
        return nil
    }

    var string: String? {
        if case let .string(v) = self { return v }
        return nil
    }

    var array: [InfoValue]? {
        if case let .array(v) = self { return v }
        return nil
    }

    var object: [String: InfoValue]? {
        if case let .object(v) = self { return v }
        return nil
    }

    var sendable: (any Sendable)? {
        if case let .sendable(v) = self { return v }
        return nil
    }

    func cast<T>(_ type: T.Type) -> T? {
        switch self {
        case let .bool(v):
            return v as? T
        case let .int(v):
            return v as? T
        case let .double(v):
            return v as? T
        case let .string(v):
            return v as? T
        case let .array(v):
            return v as? T
        case let .object(v):
            return v as? T
        case let .sendable(v):
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
