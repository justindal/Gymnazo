public protocol ValueSpace<Value>: Sendable {
    associatedtype Value: Sendable

    func containsValue(_ value: Value) -> Bool
    func sample(using generator: inout SeededGenerator) throws -> Value
}

extension ValueSpace {
    public func sample(using generator: inout SeededGenerator) throws -> Value {
        throw GymnazoError.unsupportedSpaceOperation(
            operation: "sample",
            spaceType: String(describing: Self.self)
        )
    }
}

public struct ValueBatch<Value: Sendable>: Sendable {
    public let values: [Value]

    public init(_ values: [Value]) {
        self.values = values
    }

    public var count: Int {
        values.count
    }
}

public protocol ValueBatchingSpace: ValueSpace {
    func batch(_ values: [Value]) throws -> ValueBatch<Value>
    func unbatch(_ batch: ValueBatch<Value>) throws -> [Value]
}

extension ValueBatchingSpace {
    public func batch(_ values: [Value]) throws -> ValueBatch<Value> {
        guard values.allSatisfy(containsValue) else {
            throw GymnazoError.invalidState("Cannot batch a value outside the space.")
        }
        return ValueBatch(values)
    }

    public func unbatch(_ batch: ValueBatch<Value>) throws -> [Value] {
        guard batch.values.allSatisfy(containsValue) else {
            throw GymnazoError.invalidState("Cannot unbatch a value outside the space.")
        }
        return batch.values
    }
}
