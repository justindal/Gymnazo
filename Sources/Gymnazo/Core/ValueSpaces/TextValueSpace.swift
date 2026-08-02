public struct TextValueSpace: ValueBatchingSpace {
    public let minLength: Int
    public let maxLength: Int
    public let charset: Set<Character>
    private let characters: [Character]

    public init(minLength: Int, maxLength: Int, charset: [Character]) throws {
        guard minLength >= 0, maxLength >= minLength, !charset.isEmpty else {
            throw GymnazoError.invalidConfiguration("Text space bounds and charset are invalid.")
        }
        self.minLength = minLength
        self.maxLength = maxLength
        self.charset = Set(charset)
        self.characters = Array(Set(charset)).sorted { String($0) < String($1) }
    }

    public func containsValue(_ value: String) -> Bool {
        (minLength...maxLength).contains(value.count) && value.allSatisfy(charset.contains)
    }

    public func sample(using generator: inout SeededGenerator) throws -> String {
        let length = Int.random(in: minLength...maxLength, using: &generator)
        return String(
            (0..<length).map { _ in
                characters[Int.random(in: characters.indices, using: &generator)]
            })
    }
}

extension TextValueSpace: ValueFlatteningSpace {
    public var flatDimension: Int? {
        maxLength
    }

    public func flattenValue(_ value: String) throws -> FlattenedValue {
        guard containsValue(value) else {
            throw GymnazoError.invalidState("Cannot flatten a value outside the text space.")
        }
        var flattened = value.map { character in
            Float(characters.firstIndex(of: character) ?? characters.count)
        }
        flattened.append(
            contentsOf: [Float](
                repeating: Float(characters.count), count: maxLength - flattened.count)
        )
        return .vector(flattened)
    }

    public func unflattenValue(_ value: FlattenedValue) throws -> String {
        guard case .vector(let values) = value, values.count == maxLength else {
            throw GymnazoError.invalidState("The flattened text value has an invalid shape.")
        }
        var result = ""
        var reachedPadding = false
        for value in values {
            guard value.isFinite, value.rounded() == value else {
                throw GymnazoError.invalidState("Flattened text indices must be finite integers.")
            }
            let index = Int(value)
            if index == characters.count {
                reachedPadding = true
                continue
            }
            guard !reachedPadding else {
                throw GymnazoError.invalidState("Flattened text padding must form a suffix.")
            }
            guard characters.indices.contains(index) else {
                throw GymnazoError.invalidState("A flattened text index is outside the charset.")
            }
            result.append(characters[index])
        }
        guard containsValue(result) else {
            throw GymnazoError.invalidState("The flattened text value is outside the space.")
        }
        return result
    }
}
