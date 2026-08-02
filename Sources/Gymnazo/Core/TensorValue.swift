public struct TensorValue<Element: Sendable & Equatable>: Sendable, Equatable {
    public let elements: [Element]
    public let shape: [Int]

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

    public var count: Int {
        elements.count
    }

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
