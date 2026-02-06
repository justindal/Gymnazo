import MLX

/// A space representing strings composed from a fixed character set.
///
/// A sample is a `String` whose length is in `[minLength, maxLength]` and whose characters all belong to `charset`.
/// Note: This space samples character indices as MLXArray for Space protocol conformance.
/// Use `sampleString` for direct String sampling.
public struct TextSpace: Space {
    public let minLength: Int
    public let maxLength: Int
    public let charset: [Character]

    public var shape: [Int]? { [maxLength] }
    public var dtype: DType? { .int32 }

    /// Creates a text space with an explicit character set.
    ///
    /// - Parameters:
    ///   - minLength: Minimum string length (inclusive).
    ///   - maxLength: Maximum string length (inclusive).
    ///   - charset: Allowed characters.
    public init(minLength: Int, maxLength: Int, charset: [Character]) {
        precondition(minLength >= 0, "minLength must be non-negative")
        precondition(maxLength >= minLength, "maxLength must be >= minLength")
        precondition(!charset.isEmpty, "charset must be non-empty")
        self.minLength = minLength
        self.maxLength = maxLength
        self.charset = charset
    }

    /// Creates a text space with a default alphanumeric character set.
    ///
    /// - Parameters:
    ///   - minLength: Minimum string length (inclusive).
    ///   - maxLength: Maximum string length (inclusive).
    public init(minLength: Int, maxLength: Int) {
        let defaultChars = Array("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".map { $0 })
        self.init(minLength: minLength, maxLength: maxLength, charset: defaultChars)
    }

    /// Samples character indices as an MLXArray (padded to maxLength with -1).
    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> MLXArray {
        let keys = MLX.split(key: key, into: 2)
        let lenKey = keys[0]
        let charKey = keys[1]

        let length = Int(MLX.randInt(low: minLength, high: maxLength + 1, key: lenKey).item(Int32.self))

        if length == 0 {
            return MLXArray([Int32](repeating: -1, count: maxLength))
        }

        let rand = MLX.uniform(low: 0, high: 1, [length], key: charKey).asType(.float32)
        let idx = (rand * Float(charset.count)).asType(.int32)
        
        if length < maxLength {
            let padding = MLXArray([Int32](repeating: -1, count: maxLength - length))
            return MLX.concatenated([idx, padding])
        }
        return idx
    }

    /// Samples a random string from the space.
    public func sampleString(key: MLXArray, mask: MLXArray? = nil, probability: MLXArray? = nil) -> String {
        let keys = MLX.split(key: key, into: 2)
        let lenKey = keys[0]
        let charKey = keys[1]

        let length = Int(MLX.randInt(low: minLength, high: maxLength + 1, key: lenKey).item(Int32.self))

        if length == 0 {
            return ""
        }

        let rand = MLX.uniform(low: 0, high: 1, [length], key: charKey).asType(.float32)
        let idx = (rand * Float(charset.count)).asType(.int32).asArray(Int32.self)

        var out: [Character] = []
        out.reserveCapacity(length)
        for i in 0..<length {
            let j = Int(idx[i])
            out.append(charset[j])
        }
        return String(out)
    }

    /// Returns `true` if the array represents valid character indices.
    public func contains(_ x: MLXArray) -> Bool {
        if x.shape != [maxLength] { return false }
        let vals = x.asArray(Int32.self)
        var validCount = 0
        for v in vals {
            if v >= 0 && v < Int32(charset.count) {
                validCount += 1
            } else if v != -1 {
                return false
            }
        }
        return validCount >= minLength && validCount <= maxLength
    }

    /// Returns `true` if the string length is in range and all characters are members of `charset`.
    public func containsString(_ x: String) -> Bool {
        let len = x.count
        if len < minLength || len > maxLength { return false }
        let allowed = Set(charset)
        for ch in x {
            if !allowed.contains(ch) { return false }
        }
        return true
    }
}

