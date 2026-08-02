public struct MultiDiscreteValueSpace: ValueBatchingSpace {
    public let nvec: TensorValue<Int>
    public let start: TensorValue<Int>

    public init(nvec: TensorValue<Int>, start: TensorValue<Int>? = nil) throws {
        let start =
            try start ?? TensorValue([Int](repeating: 0, count: nvec.count), shape: nvec.shape)
        guard nvec.shape == start.shape else {
            throw GymnazoError.invalidConfiguration("MultiDiscrete bounds must have equal shapes.")
        }
        guard nvec.elements.allSatisfy({ $0 > 0 }) else {
            throw GymnazoError.invalidConfiguration("MultiDiscrete counts must be positive.")
        }
        for (offset, count) in zip(start.elements, nvec.elements) {
            guard !offset.addingReportingOverflow(count).overflow else {
                throw GymnazoError.invalidConfiguration("MultiDiscrete bounds overflow Int.")
            }
        }
        self.nvec = nvec
        self.start = start
    }

    public init(_ nvec: [Int], start: [Int]? = nil) throws {
        let counts = try TensorValue(nvec, shape: [nvec.count])
        let offsets = try start.map { try TensorValue($0, shape: [nvec.count]) }
        try self.init(nvec: counts, start: offsets)
    }

    public var shape: [Int] {
        nvec.shape
    }

    public func containsValue(_ value: TensorValue<Int>) -> Bool {
        guard value.shape == shape else { return false }
        return zip(zip(value.elements, start.elements), nvec.elements).allSatisfy { pair, count in
            let (element, offset) = pair
            let upper = offset.addingReportingOverflow(count)
            return !upper.overflow && element >= offset && element < upper.partialValue
        }
    }

    public func sample(using generator: inout SeededGenerator) throws
        -> TensorValue<Int>
    {
        let values = try zip(start.elements, nvec.elements).map { offset, count in
            let upper = offset.addingReportingOverflow(count)
            guard !upper.overflow else {
                throw GymnazoError.invalidConfiguration("MultiDiscrete bounds overflow Int.")
            }
            return Int.random(in: offset..<upper.partialValue, using: &generator)
        }
        return try TensorValue(values, shape: shape)
    }
}
