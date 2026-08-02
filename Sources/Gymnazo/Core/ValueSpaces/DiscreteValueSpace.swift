/// A finite integer space with optional non-zero starting offset.
public struct DiscreteValueSpace: ValueBatchingSpace {
    public let n: Int
    public let start: Int

    /// Creates n consecutive integer values starting at start.
    ///
    /// - Parameters:
    ///   - n: Number of values in the range.
    ///   - start: First value in the range.
    /// - Throws: If n is not positive or the range overflows.
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

    /// Returns whether an integer is in the discrete range.
    ///
    /// - Parameter value: The integer to validate.
    /// - Returns: True when the integer is in the range.
    public func containsValue(_ value: Int) -> Bool {
        let upperBound = start.addingReportingOverflow(n)
        guard !upperBound.overflow else { return false }
        return (start..<upperBound.partialValue).contains(value)
    }

    /// Samples one integer using the supplied generator.
    ///
    /// - Parameter generator: The random generator used for sampling.
    /// - Returns: A sampled integer in the range.
    /// - Throws: If the range cannot be represented safely.
    public func sample(using generator: inout SeededGenerator) throws -> Int {
        let upperBound = start.addingReportingOverflow(n)
        guard !upperBound.overflow else {
            throw GymnazoError.invalidConfiguration("Discrete space bounds overflow Int.")
        }
        return Int.random(in: start..<upperBound.partialValue, using: &generator)
    }
}
