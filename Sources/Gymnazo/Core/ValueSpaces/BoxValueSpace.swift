import Foundation

/// A continuous typed tensor space with per-element bounds.
public struct BoxValueSpace: ValueBatchingSpace {
    public let low: TensorValue<Float>
    public let high: TensorValue<Float>

    /// Creates a box with scalar bounds broadcast across shape.
    ///
    /// - Parameters:
    ///   - low: Lower bound applied to every element.
    ///   - high: Upper bound applied to every element.
    ///   - shape: Tensor dimensions.
    /// - Throws: If the shape or bounds are invalid.
    public init(low: Float, high: Float, shape: [Int]) throws {
        let count = try TensorValue<Float>.elementCount(for: shape)
        try self.init(
            low: TensorValue([Float](repeating: low, count: count), shape: shape),
            high: TensorValue([Float](repeating: high, count: count), shape: shape)
        )
    }

    /// Creates a box from equally shaped lower and upper tensors.
    ///
    /// - Parameters:
    ///   - low: Lower bounds.
    ///   - high: Upper bounds.
    /// - Throws: If shapes differ or any bound is invalid.
    public init(low: TensorValue<Float>, high: TensorValue<Float>) throws {
        guard low.shape == high.shape else {
            throw GymnazoError.invalidConfiguration("Box bounds must have equal shapes.")
        }
        guard
            zip(low.elements, high.elements).allSatisfy({ lower, upper in
                !lower.isNaN && !upper.isNaN && lower <= upper
            })
        else {
            throw GymnazoError.invalidConfiguration("Box bounds must be ordered and cannot be NaN.")
        }
        self.low = low
        self.high = high
    }

    public var shape: [Int] {
        low.shape
    }

    /// Returns whether a tensor has the declared shape and bounds.
    ///
    /// - Parameter value: The tensor to validate.
    /// - Returns: True when the tensor belongs to the space.
    public func containsValue(_ value: TensorValue<Float>) -> Bool {
        guard value.shape == shape else { return false }
        return zip(zip(value.elements, low.elements), high.elements).allSatisfy { pair, upper in
            let (element, lower) = pair
            return !element.isNaN && element >= lower && element <= upper
        }
    }

    /// Samples a bounded or unbounded tensor using Swift randomness.
    ///
    /// - Parameter generator: The random generator used for sampling.
    /// - Returns: A tensor in the space.
    /// - Throws: If the space configuration cannot be sampled.
    public func sample(using generator: inout SeededGenerator) throws
        -> TensorValue<Float>
    {
        var values: [Float] = []
        values.reserveCapacity(low.count)
        for (lower, upper) in zip(low.elements, high.elements) {
            values.append(sample(lower: lower, upper: upper, using: &generator))
        }
        return try TensorValue(values, shape: shape)
    }

    private func sample(
        lower: Float,
        upper: Float,
        using generator: inout SeededGenerator
    ) -> Float {
        switch (lower > -.infinity, upper < .infinity) {
        case (true, true):
            return generator.uniform(in: lower...upper)
        case (true, false):
            return lower + exponential(using: &generator)
        case (false, true):
            return upper - exponential(using: &generator)
        case (false, false):
            return normal(using: &generator)
        }
    }

    private func exponential(using generator: inout SeededGenerator) -> Float {
        let uniform = min(
            max(generator.uniform(in: 0...1), Float.leastNonzeroMagnitude),
            1 - Float.ulpOfOne
        )
        return -log(1 - uniform)
    }

    private func normal(using generator: inout SeededGenerator) -> Float {
        let first = max(
            generator.uniform(in: 0...1),
            Float.leastNonzeroMagnitude
        )
        let second = generator.uniform(in: 0...1)
        return sqrt(-2 * log(first)) * cos(2 * .pi * second)
    }
}
