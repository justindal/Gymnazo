import MLX

/// Gymnazo ``Space`` Information
/// Get metadata to rebuild spaces from checkpoint metadata
/// when loading algorithms without requiring a live environment instance.
public struct SpaceDescriptor: Codable, Sendable, Equatable {
    /// Spaces that can be serialized and reconstructed.
    public enum Kind: String, Codable, Sendable {
        case box
        case discrete
        case multiDiscrete
        case multiBinary
        case text
        case tuple
        case dict
    }

    /// Encoded space family.
    public let kind: Kind
    public let shape: [Int]?
    public let low: [Float]?
    public let high: [Float]?
    public let dtype: String?
    public let n: Int?
    public let start: Int?
    public let nvec: [Int]?
    public let minLength: Int?
    public let maxLength: Int?
    public let charset: String?
    public let tupleItems: [SpaceDescriptor]?
    public let dictItems: [String: SpaceDescriptor]?

    public init(
        kind: Kind,
        shape: [Int]? = nil,
        low: [Float]? = nil,
        high: [Float]? = nil,
        dtype: String? = nil,
        n: Int? = nil,
        start: Int? = nil,
        nvec: [Int]? = nil,
        minLength: Int? = nil,
        maxLength: Int? = nil,
        charset: String? = nil,
        tupleItems: [SpaceDescriptor]? = nil,
        dictItems: [String: SpaceDescriptor]? = nil
    ) {
        self.kind = kind
        self.shape = shape
        self.low = low
        self.high = high
        self.dtype = dtype
        self.n = n
        self.start = start
        self.nvec = nvec
        self.minLength = minLength
        self.maxLength = maxLength
        self.charset = charset
        self.tupleItems = tupleItems
        self.dictItems = dictItems
    }

    /// Builds a descriptor from a concrete Gymnazo ``Space`` value.
    public static func from(space: any Space) throws -> SpaceDescriptor {
        if let erased = space as? AnySpace {
            return try from(space: erased.base)
        }
        if let box = boxSpace(from: space) {
            let boxShape = box.shape ?? box.low.shape
            let d = try encodeDType(box.dtype ?? .float32)
            let low = box.low.asType(.float32).reshaped([-1]).asArray(Float.self)
            let high = box.high.asType(.float32).reshaped([-1]).asArray(Float.self)
            return SpaceDescriptor(
                kind: .box,
                shape: boxShape,
                low: low,
                high: high,
                dtype: d
            )
        }
        if let discrete = space as? Discrete {
            return SpaceDescriptor(
                kind: .discrete,
                n: discrete.n,
                start: discrete.start
            )
        }
        if let multiDiscrete = space as? MultiDiscrete {
            let nvecShape = multiDiscrete.shape
            let nvec = multiDiscrete.nvec.asType(.int32).reshaped([-1]).asArray(Int32.self).map {
                Int($0)
            }
            return SpaceDescriptor(kind: .multiDiscrete, shape: nvecShape, nvec: nvec)
        }
        if let multiBinary = space as? MultiBinary {
            guard let shape = multiBinary.shape else {
                throw PersistenceError.invalidCheckpoint(
                    "Cannot serialize MultiBinary without shape"
                )
            }
            return SpaceDescriptor(kind: .multiBinary, shape: shape)
        }
        if let text = space as? TextSpace {
            return SpaceDescriptor(
                kind: .text,
                minLength: text.minLength,
                maxLength: text.maxLength,
                charset: String(text.charset)
            )
        }
        if let tuple = space as? Tuple {
            let items = try tuple.spaces.map { try from(space: $0) }
            return SpaceDescriptor(kind: .tuple, tupleItems: items)
        }
        if let dict = space as? Dict {
            var items: [String: SpaceDescriptor] = [:]
            for (key, value) in dict.spaces {
                items[key] = try from(space: value)
            }
            return SpaceDescriptor(kind: .dict, dictItems: items)
        }
        throw PersistenceError.invalidCheckpoint(
            "Unsupported space for serialization: \(type(of: space))"
        )
    }

    /// Reconstructs a concrete Gymnazo ``Space`` from this descriptor.
    public func makeSpace() throws -> any Space {
        switch kind {
        case .box:
            guard let shape, let low, let high else {
                throw PersistenceError.invalidCheckpoint("Invalid Box descriptor")
            }
            guard low.count == high.count else {
                throw PersistenceError.invalidCheckpoint("Mismatched Box low/high lengths")
            }
            let expectedCount = shape.reduce(1, *)
            guard low.count == expectedCount else {
                throw PersistenceError.invalidCheckpoint("Invalid Box descriptor shape")
            }
            let targetDType = try decodeDType(dtype ?? "float32")
            let lowArray = MLXArray(low).reshaped(shape).asType(targetDType)
            let highArray = MLXArray(high).reshaped(shape).asType(targetDType)
            return Box(low: lowArray, high: highArray, dtype: targetDType)
        case .discrete:
            guard let n else {
                throw PersistenceError.invalidCheckpoint("Invalid Discrete descriptor")
            }
            return Discrete(n: n, start: start ?? 0)
        case .multiDiscrete:
            guard let nvec else {
                throw PersistenceError.invalidCheckpoint("Invalid MultiDiscrete descriptor")
            }
            if let shape {
                let expectedCount = shape.reduce(1, *)
                guard expectedCount == nvec.count else {
                    throw PersistenceError.invalidCheckpoint(
                        "Invalid MultiDiscrete descriptor shape"
                    )
                }
                let nvecArray = MLXArray(nvec).reshaped(shape)
                return MultiDiscrete(nvecArray)
            }
            return MultiDiscrete(nvec)
        case .multiBinary:
            guard let shape else {
                throw PersistenceError.invalidCheckpoint("Invalid MultiBinary descriptor")
            }
            return MultiBinary(shape: shape)
        case .text:
            guard let minLength, let maxLength, let charset else {
                throw PersistenceError.invalidCheckpoint("Invalid Text descriptor")
            }
            return TextSpace(
                minLength: minLength,
                maxLength: maxLength,
                charset: Array(charset)
            )
        case .tuple:
            guard let tupleItems else {
                throw PersistenceError.invalidCheckpoint("Invalid Tuple descriptor")
            }
            var spaces: [any Space] = []
            spaces.reserveCapacity(tupleItems.count)
            for item in tupleItems {
                spaces.append(try item.makeSpace())
            }
            return Tuple(spaces)
        case .dict:
            guard let dictItems else {
                throw PersistenceError.invalidCheckpoint("Invalid Dict descriptor")
            }
            var spaces: [String: any Space] = [:]
            for (key, descriptor) in dictItems {
                spaces[key] = try descriptor.makeSpace()
            }
            return Dict(spaces)
        }
    }
}

private func encodeDType(_ dtype: DType) throws -> String {
    if dtype == .float32 { return "float32" }
    if dtype == .float16 { return "float16" }
    if dtype == .bfloat16 { return "bfloat16" }
    if dtype == .float64 { return "float64" }
    if dtype == .int32 { return "int32" }
    if dtype == .int64 { return "int64" }
    if dtype == .uint8 { return "uint8" }
    if dtype == .bool { return "bool" }
    throw PersistenceError.invalidCheckpoint("Unsupported dtype: \(dtype)")
}

private func decodeDType(_ rawValue: String) throws -> DType {
    switch rawValue {
    case "float16": return .float16
    case "bfloat16": return .bfloat16
    case "float32": return .float32
    case "float64": return .float64
    case "int32": return .int32
    case "int64": return .int64
    case "uint8": return .uint8
    case "bool": return .bool
    default:
        throw PersistenceError.invalidCheckpoint("Unsupported dtype raw value: \(rawValue)")
    }
}
