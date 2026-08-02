public struct MultiBinaryValueSpace: ValueBatchingSpace {
    public let shape: [Int]

    public init(shape: [Int]) throws {
        _ = try TensorValue<Bool>.elementCount(for: shape)
        self.shape = shape
    }

    public init(n: Int) throws {
        try self.init(shape: [n])
    }

    public func containsValue(_ value: TensorValue<Bool>) -> Bool {
        value.shape == shape
    }

    public func sample(using generator: inout SeededGenerator) throws
        -> TensorValue<Bool>
    {
        let count = try TensorValue<Bool>.elementCount(for: shape)
        let values = (0..<count).map { _ in generator.next() & 1 == 1 }
        return try TensorValue(values, shape: shape)
    }
}
