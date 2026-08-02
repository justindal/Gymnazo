/// A shaped, sendable tensor value stored in Swift memory.
public struct TensorValue<Element: Sendable & Equatable>: Sendable, Equatable {
    public let elements: [Element]
    public let shape: [Int]

    /// Creates a tensor after validating that the shape contains all elements.
    ///
    /// - Parameters:
    ///   - elements: Scalar values in row-major order.
    ///   - shape: The tensor dimensions.
    /// - Throws: If the shape is invalid or its element count differs from `elements`.
    public init(_ elements: [Element], shape: [Int]) throws {
        let count = try Self.elementCount(for: shape)
        guard elements.count == count else {
            throw GymnazoError.invalidConfiguration(
                "Tensor value shape \(shape) requires \(count) elements, received \(elements.count)."
            )
        }
        self.elements = elements
        self.shape = shape
    }

    /// The number of scalar elements in the tensor.
    public var count: Int {
        elements.count
    }

    /// Returns the number of scalar elements required by a shape.
    ///
    /// - Parameter shape: The tensor dimensions.
    /// - Returns: The product of the dimensions.
    /// - Throws: If a dimension is negative or the product overflows.
    public static func elementCount(for shape: [Int]) throws -> Int {
        var count = 1
        for dimension in shape {
            guard dimension >= 0 else {
                throw GymnazoError.invalidConfiguration(
                    "Tensor value dimensions must be non-negative.")
            }
            let result = count.multipliedReportingOverflow(by: dimension)
            guard !result.overflow else {
                throw GymnazoError.invalidConfiguration("Tensor value shape is too large.")
            }
            count = result.partialValue
        }
        return count
    }
}
