public struct SeededGenerator: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }

    public mutating func uniform(in range: ClosedRange<Float>) -> Float {
        let unit = Float(next() >> 40) / Float(1 << 24)
        return range.lowerBound + unit * (range.upperBound - range.lowerBound)
    }

    public mutating func weightedIndex(_ weights: [Double]) -> Int? {
        guard !weights.isEmpty, weights.allSatisfy({ $0 >= 0 && $0.isFinite }) else {
            return nil
        }
        let total = weights.reduce(0, +)
        guard total > 0, total.isFinite else { return nil }
        let threshold = Double(uniform(in: 0...1)) * total
        var cumulative = 0.0
        for (index, weight) in weights.enumerated() {
            cumulative += weight
            if threshold < cumulative {
                return index
            }
        }
        return weights.index(before: weights.endIndex)
    }
}
