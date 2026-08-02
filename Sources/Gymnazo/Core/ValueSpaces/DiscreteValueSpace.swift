public struct DiscreteValueSpace: ValueBatchingSpace {
    public let n: Int
    public let start: Int

    public init(n: Int, start: Int = 0) throws {
        guard n > 0 else {
            throw GymnazoError.invalidConfiguration("Discrete spaces require a positive count.")
        }
        guard !start.addingReportingOverflow(n).overflow else {
            throw GymnazoError.invalidConfiguration("Discrete space bounds overflow Int.")
        }
        self.n = n
        self.start = start
    }

    public func containsValue(_ value: Int) -> Bool {
        let upperBound = start.addingReportingOverflow(n)
        guard !upperBound.overflow else { return false }
        return (start..<upperBound.partialValue).contains(value)
    }

    public func sample(using generator: inout SeededGenerator) throws -> Int {
        let upperBound = start.addingReportingOverflow(n)
        guard !upperBound.overflow else {
            throw GymnazoError.invalidConfiguration("Discrete space bounds overflow Int.")
        }
        return Int.random(in: start..<upperBound.partialValue, using: &generator)
    }
}
