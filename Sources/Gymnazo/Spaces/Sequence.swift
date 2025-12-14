import MLX

public struct SequenceSample {
    public let values: MLXArray
    public let mask: MLXArray

    public init(values: MLXArray, mask: MLXArray) {
        self.values = values
        self.mask = mask
    }
}

public struct SequenceSpace<Inner: MLXSpace>: Space {
    public typealias T = SequenceSample

    public let space: Inner
    public let minLength: Int
    public let maxLength: Int

    private let elementShape: [Int]
    private let elementCount: Int

    public init(space: Inner, minLength: Int = 0, maxLength: Int) {
        precondition(minLength >= 0, "minLength must be non-negative")
        precondition(maxLength >= minLength, "maxLength must be >= minLength")
        guard let shape = space.shape else {
            fatalError("SequenceSpace requires an inner space with a defined shape")
        }
        self.space = space
        self.minLength = minLength
        self.maxLength = maxLength
        self.elementShape = shape
        self.elementCount = shape.reduce(1, *)
    }

    public var shape: [Int]? { nil }
    public var dtype: DType? { nil }

    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> SequenceSample {
        let keys = MLX.split(key: key, into: 2)
        let lenKey = keys[0]
        let valKey = keys[1]

        let length = Int(MLX.randInt(low: minLength, high: maxLength + 1, key: lenKey).item(Int32.self))
        let values = space.sampleBatch(key: valKey, count: maxLength)

        var maskI32 = [Int32](repeating: 0, count: maxLength)
        if length > 0 {
            for i in 0..<min(length, maxLength) {
                maskI32[i] = 1
            }
        }
        let maskArr = MLXArray(maskI32).asType(.bool)
        return SequenceSample(values: values, mask: maskArr)
    }

    public func contains(_ x: SequenceSample) -> Bool {
        if x.mask.shape != [maxLength] { return false }
        if x.values.shape.count < 1 { return false }
        if x.values.shape[0] != maxLength { return false }
        if Array(x.values.shape.dropFirst()) != elementShape { return false }

        let maskVals = x.mask.asType(.bool).asArray(Bool.self)
        var prefixCount = 0
        while prefixCount < maskVals.count && maskVals[prefixCount] {
            prefixCount += 1
        }
        for i in prefixCount..<maskVals.count {
            if maskVals[i] { return false }
        }

        let d = space.dtype ?? .float32
        if d != .float32 && d != .int32 {
            return false
        }

        let flat = x.values.asType(d).reshaped([maxLength, elementCount])

        if d == .float32 {
            let allVals = flat.asArray(Float.self)
            for i in 0..<prefixCount {
                let start = i * elementCount
                let end = start + elementCount
                let row = MLXArray(Array(allVals[start..<end])).reshaped(elementShape).asType(.float32)
                if !space.contains(row) { return false }
            }
            return true
        }

        let allVals = flat.asArray(Int32.self)
        for i in 0..<prefixCount {
            let start = i * elementCount
            let end = start + elementCount
            let row = MLXArray(Array(allVals[start..<end])).reshaped(elementShape).asType(.int32)
            if !space.contains(row) { return false }
        }
        return true
    }
}

