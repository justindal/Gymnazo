import Foundation
import MLX

/// Returns the name of a space.
///
/// - Parameter space: The space to get the name of.
/// - Returns: The name of the space.
private func spaceTypeName(_ space: any Space) -> String {
    String(describing: type(of: space))
}

/// Creates a ``GymnazoError.invalidSampleType`` error.
///
/// - Parameters:
///   - operation: The operation that caused the error.
///   - expected: The expected type of the sample.
///   - actual: The actual type of the sample.
/// - Returns: A ``GymnazoError.invalidSampleType`` error.
private func makeSampleTypeError(
    operation: String,
    expected: String,
    actual: Any
) -> GymnazoError {
    GymnazoError.invalidSampleType(
        operation: operation,
        expected: expected,
        actual: String(describing: type(of: actual))
    )
}

/// Returns a space equivalent to `space` with its leaf components flattened where possible.
///
/// Spaces like ``Box``, ``Discrete``, ``MultiBinary``, ``MultiDiscrete``, ``Tuple``, ``Dict``, and ``TextSpace`` flatten to a single ``Box``.
/// - ``SequenceSpace`` and ``Graph`` preserve their structure and flatten their element/node/edge spaces.
public func flatten_space(_ space: any Space) throws -> any Space {
    if let box = space as? Box {
        let (low, high, dtype) = flattenBoxBounds(box)
        let lowArr = MLXArray(low).asType(dtype)
        let highArr = MLXArray(high).asType(dtype)
        return Box(low: lowArr, high: highArr, dtype: dtype)
    }

    if space is Discrete || space is MultiDiscrete || space is MultiBinary || space is Tuple
        || space is Dict || space is TextSpace
    {
        let (low, high, dtype) = try flattenBoxBounds(space)
        let lowArr = MLXArray(low).asType(dtype)
        let highArr = MLXArray(high).asType(dtype)
        return Box(low: lowArr, high: highArr, dtype: dtype)
    }

    if let seq = space as? any AnySequenceSpace {
        let flattenedElement = try flatten_space(seq.elementSpace)
        guard let elementBox = flattenedElement as? Box else {
            throw GymnazoError.unsupportedSpaceOperation(
                operation: "flatten_space.sequence_element",
                spaceType: spaceTypeName(seq.elementSpace)
            )
        }
        return SequenceSpace(space: elementBox, minLength: seq.minLength, maxLength: seq.maxLength)
    }

    if let graph = space as? any AnyGraphSpace {
        let flattenedNode = try flatten_space(graph.nodeSpaceAny)
        let flattenedEdge = try flatten_space(graph.edgeSpaceAny)
        guard let nodeBox = flattenedNode as? Box else {
            throw GymnazoError.unsupportedSpaceOperation(
                operation: "flatten_space.graph_node",
                spaceType: spaceTypeName(graph.nodeSpaceAny)
            )
        }
        guard let edgeBox = flattenedEdge as? Box else {
            throw GymnazoError.unsupportedSpaceOperation(
                operation: "flatten_space.graph_edge",
                spaceType: spaceTypeName(graph.edgeSpaceAny)
            )
        }
        return Graph(
            nodeSpace: nodeBox,
            edgeSpace: edgeBox,
            maxNodes: graph.maxNodes,
            maxEdges: graph.maxEdges,
            allowSelfLoops: graph.allowSelfLoops,
            directed: graph.directed
        )
    }

    throw GymnazoError.unsupportedSpaceOperation(
        operation: "flatten_space",
        spaceType: spaceTypeName(space)
    )
}

/// Returns a flattened ``Box`` if and only if `flatten_space(space)` is a ``Box``.
public func flattenSpaceToBox(_ space: any Space) throws -> Box? {
    try flatten_space(space) as? Box
}

/// Returns the dimensionality of a space that flattens to a ``Box``.
///
/// - Note: This is only defined when `flatten_space(space)` returns a ``Box``.
public func flatdim(_ space: any Space) throws -> Int {
    guard let box = try flattenSpaceToBox(space) else {
        throw GymnazoError.unsupportedSpaceOperation(
            operation: "flatdim",
            spaceType: spaceTypeName(space)
        )
    }
    guard let shape = box.shape else { return 0 }
    return shape.reduce(1, *)
}

/// Flattens a sample from `space` into its Gymnasium-style flattened representation.
///
/// - Returns:
///   - `MLXArray` when `flatten_space(space)` is a ``Box``.
///   - ``SequenceSample`` when `space` is a ``SequenceSpace``.
///   - ``GraphSample`` when `space` is a ``Graph``.
public func flatten(space: any Space, sample: Any) throws -> Any {
    if try flattenSpaceToBox(space) != nil {
        return try flattenToBox(space: space, sample: sample)
    }

    if space is any AnySequenceSpace {
        guard let seqSample = sample as? SequenceSample else {
            throw makeSampleTypeError(
                operation: "flatten.sequence",
                expected: String(describing: SequenceSample.self),
                actual: sample
            )
        }
        return try flattenSequence(space: space, sample: seqSample)
    }

    if space is any AnyGraphSpace {
        guard let gSample = sample as? GraphSample else {
            throw makeSampleTypeError(
                operation: "flatten.graph",
                expected: String(describing: GraphSample.self),
                actual: sample
            )
        }
        return try flattenGraph(space: space, sample: gSample)
    }

    throw GymnazoError.unsupportedSpaceOperation(
        operation: "flatten",
        spaceType: spaceTypeName(space)
    )
}

/// Inverse of ``flatten(space:sample:)``.
public func unflatten(space: any Space, flattened: Any) throws -> Any {
    if try flattenSpaceToBox(space) != nil {
        guard let flat = flattened as? MLXArray else {
            throw makeSampleTypeError(
                operation: "unflatten.box",
                expected: String(describing: MLXArray.self),
                actual: flattened
            )
        }
        return try unflattenFromBox(space: space, flat: flat)
    }

    if let seq = space as? any AnySequenceSpace {
        guard let seqSample = flattened as? SequenceSample else {
            throw makeSampleTypeError(
                operation: "unflatten.sequence",
                expected: String(describing: SequenceSample.self),
                actual: flattened
            )
        }
        return unflattenSequence(space: seq, flattened: seqSample)
    }

    if let graph = space as? any AnyGraphSpace {
        guard let gSample = flattened as? GraphSample else {
            throw makeSampleTypeError(
                operation: "unflatten.graph",
                expected: String(describing: GraphSample.self),
                actual: flattened
            )
        }
        return unflattenGraph(space: graph, flattened: gSample)
    }

    throw GymnazoError.unsupportedSpaceOperation(
        operation: "unflatten",
        spaceType: spaceTypeName(space)
    )
}

private func flattenBoxBounds(_ space: any Space) throws -> (
    low: [Float], high: [Float], dtype: DType
) {
    if let box = space as? Box {
        return flattenBoxBounds(box)
    }

    if let discrete = space as? Discrete {
        let n = discrete.n
        return (
            low: [Float](repeating: 0, count: n),
            high: [Float](repeating: 1, count: n),
            dtype: .float32
        )
    }

    if let multiDiscrete = space as? MultiDiscrete {
        let nvec = multiDiscrete.nvec.asType(.int32).asArray(Int32.self)
        let n = nvec.reduce(0) { $0 + Int($1) }
        return (
            low: [Float](repeating: 0, count: n),
            high: [Float](repeating: 1, count: n),
            dtype: .float32
        )
    }

    if let multiBinary = space as? MultiBinary {
        let n = multiBinary.shape?.reduce(1, *) ?? 0
        return (
            low: [Float](repeating: 0, count: n),
            high: [Float](repeating: 1, count: n),
            dtype: .float32
        )
    }

    if let text = space as? TextSpace {
        let n = text.maxLength
        let high = Float(text.charset.count - 1)
        return (
            low: [Float](repeating: -1, count: n),
            high: [Float](repeating: high, count: n),
            dtype: .float32
        )
    }

    if let tuple = space as? Tuple {
        var low: [Float] = []
        var high: [Float] = []
        for s in tuple.spaces {
            let b = try flattenSpaceToBox(s)
            guard let box = b else {
                throw GymnazoError.unsupportedSpaceOperation(
                    operation: "flattenBoxBounds.tuple_element",
                    spaceType: spaceTypeName(s)
                )
            }
            let (l, h, _) = flattenBoxBounds(box)
            low.append(contentsOf: l)
            high.append(contentsOf: h)
        }
        return (low: low, high: high, dtype: .float32)
    }

    if let dict = space as? Dict {
        var low: [Float] = []
        var high: [Float] = []
        let keys = dict.spaces.keys.sorted()
        for k in keys {
            let s = dict.spaces[k]!
            let b = try flattenSpaceToBox(s)
            guard let box = b else {
                throw GymnazoError.unsupportedSpaceOperation(
                    operation: "flattenBoxBounds.dict_value",
                    spaceType: spaceTypeName(s)
                )
            }
            let (l, h, _) = flattenBoxBounds(box)
            low.append(contentsOf: l)
            high.append(contentsOf: h)
        }
        return (low: low, high: high, dtype: .float32)
    }

    throw GymnazoError.unsupportedSpaceOperation(
        operation: "flattenBoxBounds",
        spaceType: spaceTypeName(space)
    )
}

private func flattenBoxBounds(_ box: Box) -> (low: [Float], high: [Float], dtype: DType) {
    let dtype = box.dtype ?? .float32
    let low = box.low.asType(dtype).reshaped([-1]).asArray(Float.self)
    let high = box.high.asType(dtype).reshaped([-1]).asArray(Float.self)
    return (low: low, high: high, dtype: dtype)
}

private func flattenToBox(space: any Space, sample: Any) throws -> MLXArray {
    if let box = space as? Box {
        guard let x = sample as? MLXArray else {
            throw makeSampleTypeError(
                operation: "flattenToBox.box",
                expected: String(describing: MLXArray.self),
                actual: sample
            )
        }
        return x.reshaped([-1]).asType(box.dtype ?? .float32)
    }

    if let discrete = space as? Discrete {
        guard let x = sample as? MLXArray else {
            throw makeSampleTypeError(
                operation: "flattenToBox.discrete",
                expected: String(describing: MLXArray.self),
                actual: sample
            )
        }
        let idx = Int(x.singletonValue(Int32.self)) - discrete.start
        var onehot = [Float](repeating: 0, count: discrete.n)
        if idx >= 0 && idx < discrete.n {
            onehot[idx] = 1
        }
        return MLXArray(onehot).asType(.float32)
    }

    if let multiDiscrete = space as? MultiDiscrete {
        guard let x = sample as? MLXArray else {
            throw makeSampleTypeError(
                operation: "flattenToBox.multiDiscrete",
                expected: String(describing: MLXArray.self),
                actual: sample
            )
        }
        let nvec = multiDiscrete.nvec.asType(.int32).asArray(Int32.self).map { Int($0) }
        let xs = x.asType(.int32).reshaped([-1]).asArray(Int32.self).map { Int($0) }
        guard xs.count == nvec.count else {
            throw GymnazoError.invalidSampleCount(
                operation: "flattenToBox.multiDiscrete",
                expected: nvec.count,
                actual: xs.count
            )
        }

        var out: [Float] = []
        out.reserveCapacity(nvec.reduce(0, +))

        for i in 0..<nvec.count {
            let n = nvec[i]
            let v = xs[i]
            for j in 0..<n {
                out.append(j == v ? 1 : 0)
            }
        }
        return MLXArray(out).asType(.float32)
    }

    if space is MultiBinary {
        guard let x = sample as? MLXArray else {
            throw makeSampleTypeError(
                operation: "flattenToBox.multiBinary",
                expected: String(describing: MLXArray.self),
                actual: sample
            )
        }
        return x.reshaped([-1]).asType(.float32)
    }

    if space is TextSpace {
        guard let x = sample as? MLXArray else {
            throw makeSampleTypeError(
                operation: "flattenToBox.text",
                expected: String(describing: MLXArray.self),
                actual: sample
            )
        }
        return x.reshaped([-1]).asType(.float32)
    }

    if let tuple = space as? Tuple {
        guard let xs = sample as? [Any] else {
            throw makeSampleTypeError(
                operation: "flattenToBox.tuple",
                expected: String(describing: [Any].self),
                actual: sample
            )
        }
        guard xs.count == tuple.spaces.count else {
            throw GymnazoError.invalidSampleCount(
                operation: "flattenToBox.tuple",
                expected: tuple.spaces.count,
                actual: xs.count
            )
        }
        var parts: [MLXArray] = []
        parts.reserveCapacity(tuple.spaces.count)
        for i in 0..<tuple.spaces.count {
            let part = try flatten(space: tuple.spaces[i], sample: xs[i])
            guard let arr = part as? MLXArray else {
                throw makeSampleTypeError(
                    operation: "flattenToBox.tuple_part",
                    expected: String(describing: MLXArray.self),
                    actual: part
                )
            }
            parts.append(arr)
        }
        return concatenateFlat(parts)
    }

    if let dict = space as? Dict {
        guard let x = sample as? [String: Any] else {
            throw makeSampleTypeError(
                operation: "flattenToBox.dict",
                expected: String(describing: [String: Any].self),
                actual: sample
            )
        }
        let keys = dict.spaces.keys.sorted()
        var parts: [MLXArray] = []
        parts.reserveCapacity(keys.count)
        for k in keys {
            guard let value = x[k] else {
                throw GymnazoError.missingSampleKey(k)
            }
            let part = try flatten(space: dict.spaces[k]!, sample: value)
            guard let arr = part as? MLXArray else {
                throw makeSampleTypeError(
                    operation: "flattenToBox.dict_value",
                    expected: String(describing: MLXArray.self),
                    actual: part
                )
            }
            parts.append(arr)
        }
        return concatenateFlat(parts)
    }

    throw GymnazoError.unsupportedSpaceOperation(
        operation: "flattenToBox",
        spaceType: spaceTypeName(space)
    )
}

private func unflattenFromBox(space: any Space, flat: MLXArray) throws -> Any {
    if let box = space as? Box {
        guard let shape = box.shape else { return flat }
        return flat.asType(box.dtype ?? .float32).reshaped(shape)
    }

    if let discrete = space as? Discrete {
        let vals = flat.asType(.float32).reshaped([-1]).asArray(Float.self)
        var best = 0
        var bestV = -Float.infinity
        for i in 0..<vals.count {
            if vals[i] > bestV {
                bestV = vals[i]
                best = i
            }
        }
        return MLXArray(Int32(discrete.start + best))
    }

    if let multiDiscrete = space as? MultiDiscrete {
        let nvec = multiDiscrete.nvec.asType(.int32).asArray(Int32.self).map { Int($0) }
        let vals = flat.asType(.float32).reshaped([-1]).asArray(Float.self)
        var out: [Int32] = []
        out.reserveCapacity(nvec.count)
        var offset = 0
        for n in nvec {
            var best = 0
            var bestV = -Float.infinity
            for j in 0..<n {
                let v = vals[offset + j]
                if v > bestV {
                    bestV = v
                    best = j
                }
            }
            out.append(Int32(best))
            offset += n
        }
        return MLXArray(out).asType(.int32)
    }

    if let multiBinary = space as? MultiBinary {
        guard let shape = multiBinary.shape else { return flat }
        let vals = flat.asType(.float32).reshaped([-1]).asArray(Float.self)
        let out = vals.map { $0 >= 0.5 ? Int32(1) : Int32(0) }
        return MLXArray(out).asType(.int32).reshaped(shape)
    }

    if let text = space as? TextSpace {
        let vals = flat.asType(.float32).reshaped([-1]).asArray(Float.self)
        var indices = [Int32](repeating: -1, count: text.maxLength)
        for i in 0..<min(vals.count, text.maxLength) {
            let v = Int(vals[i].rounded())
            if v < 0 { break }
            if v >= 0 && v < text.charset.count {
                indices[i] = Int32(v)
            }
        }
        return MLXArray(indices)
    }

    if let tuple = space as? Tuple {
        let vals = flat.asType(.float32).reshaped([-1]).asArray(Float.self)
        var offset = 0
        var out: [Any] = []
        out.reserveCapacity(tuple.spaces.count)
        for s in tuple.spaces {
            let d = try flatdim(s)
            let part = MLXArray(Array(vals[offset..<(offset + d)])).asType(.float32)
            out.append(try unflatten(space: s, flattened: part))
            offset += d
        }
        return out
    }

    if let dict = space as? Dict {
        let vals = flat.asType(.float32).reshaped([-1]).asArray(Float.self)
        let keys = dict.spaces.keys.sorted()
        var offset = 0
        var out: [String: Any] = [:]
        for k in keys {
            let s = dict.spaces[k]!
            let d = try flatdim(s)
            let part = MLXArray(Array(vals[offset..<(offset + d)])).asType(.float32)
            out[k] = try unflatten(space: s, flattened: part)
            offset += d
        }
        return out
    }

    throw GymnazoError.unsupportedSpaceOperation(
        operation: "unflattenFromBox",
        spaceType: spaceTypeName(space)
    )
}

private func flattenSequence(space: any Space, sample: SequenceSample) throws -> SequenceSample {
    guard let seq = space as? any AnySequenceSpace else {
        throw GymnazoError.unsupportedSpaceOperation(
            operation: "flattenSequence",
            spaceType: spaceTypeName(space)
        )
    }

    let elementSpace = seq.elementSpace

    if let box = elementSpace as? Box {
        let elementShape = box.shape ?? []
        let elementCount = elementShape.reduce(1, *)
        let values = sample.values.asType(.float32).reshaped([seq.maxLength, elementCount])
        return SequenceSample(values: values, mask: sample.mask)
    }

    if elementSpace is MultiBinary {
        let elementShape = elementSpace.shape ?? []
        let elementCount = elementShape.reduce(1, *)
        let values = sample.values.asType(.float32).reshaped([seq.maxLength, elementCount])
        return SequenceSample(values: values, mask: sample.mask)
    }

    if let md = elementSpace as? MultiDiscrete {
        let nvec = md.nvec.asType(.int32).asArray(Int32.self).map { Int($0) }
        let cols = nvec.count
        let total = nvec.reduce(0, +)

        let xs = sample.values.asType(.int32).reshaped([seq.maxLength * cols]).asArray(Int32.self)
            .map { Int($0) }
        var out = [Float](repeating: 0, count: seq.maxLength * total)

        for i in 0..<seq.maxLength {
            var outOffset = i * total
            let inOffset = i * cols
            for j in 0..<cols {
                let n = nvec[j]
                let v = xs[inOffset + j]
                if v >= 0 && v < n {
                    out[outOffset + v] = 1
                }
                outOffset += n
            }
        }

        let values = MLXArray(out).asType(.float32).reshaped([seq.maxLength, total])
        return SequenceSample(values: values, mask: sample.mask)
    }

    let elementShape = elementSpace.shape ?? []
    let elementCount = elementShape.reduce(1, *)
    let values = sample.values.asType(.float32).reshaped([seq.maxLength, elementCount])
    return SequenceSample(values: values, mask: sample.mask)
}

private func unflattenSequence(space: any AnySequenceSpace, flattened: SequenceSample)
    -> SequenceSample
{
    let elementSpace = space.elementSpace

    if let box = elementSpace as? Box {
        let elementShape = box.shape ?? []
        let values = flattened.values.asType(box.dtype ?? .float32).reshaped(
            [space.maxLength] + elementShape)
        return SequenceSample(values: values, mask: flattened.mask)
    }

    if let mb = elementSpace as? MultiBinary {
        let elementShape = mb.shape ?? []
        let elementCount = elementShape.reduce(1, *)
        let vals = flattened.values.asType(.float32).reshaped([space.maxLength * elementCount])
            .asArray(Float.self)
        let out = vals.map { $0 >= 0.5 ? Int32(1) : Int32(0) }
        let values = MLXArray(out).asType(.int32).reshaped([space.maxLength] + elementShape)
        return SequenceSample(values: values, mask: flattened.mask)
    }

    if let md = elementSpace as? MultiDiscrete {
        let nvec = md.nvec.asType(.int32).asArray(Int32.self).map { Int($0) }
        let cols = nvec.count
        let total = nvec.reduce(0, +)

        let vals = flattened.values.asType(.float32).reshaped([space.maxLength * total]).asArray(
            Float.self)
        var out: [Int32] = []
        out.reserveCapacity(space.maxLength * cols)

        for i in 0..<space.maxLength {
            var offset = i * total
            for n in nvec {
                var best = 0
                var bestV = -Float.infinity
                for j in 0..<n {
                    let v = vals[offset + j]
                    if v > bestV {
                        bestV = v
                        best = j
                    }
                }
                out.append(Int32(best))
                offset += n
            }
        }

        let values = MLXArray(out).asType(.int32).reshaped([space.maxLength, cols])
        return SequenceSample(values: values, mask: flattened.mask)
    }

    let elementShape = elementSpace.shape ?? []
    let values = flattened.values.asType(.float32).reshaped([space.maxLength] + elementShape)
    return SequenceSample(values: values, mask: flattened.mask)
}

private func flattenGraph(space: any Space, sample: GraphSample) throws -> GraphSample {
    guard let g = space as? any AnyGraphSpace else {
        throw GymnazoError.unsupportedSpaceOperation(
            operation: "flattenGraph",
            spaceType: spaceTypeName(space)
        )
    }

    let nodes = flattenGraphValues(space: g.nodeSpaceAny, values: sample.nodes, count: g.maxNodes)
    let edges = flattenGraphValues(space: g.edgeSpaceAny, values: sample.edges, count: g.maxEdges)

    return GraphSample(
        nodes: nodes,
        edges: edges,
        edgeLinks: sample.edgeLinks,
        nodeMask: sample.nodeMask,
        edgeMask: sample.edgeMask
    )
}

private func unflattenGraph(space: any AnyGraphSpace, flattened: GraphSample) -> GraphSample {
    let nodes = unflattenGraphValues(
        space: space.nodeSpaceAny, values: flattened.nodes, count: space.maxNodes)
    let edges = unflattenGraphValues(
        space: space.edgeSpaceAny, values: flattened.edges, count: space.maxEdges)

    return GraphSample(
        nodes: nodes,
        edges: edges,
        edgeLinks: flattened.edgeLinks,
        nodeMask: flattened.nodeMask,
        edgeMask: flattened.edgeMask
    )
}

private func flattenGraphValues(space: any TensorSpace, values: MLXArray, count: Int) -> MLXArray {
    if let box = space as? Box {
        let elementShape = box.shape ?? []
        let elementCount = elementShape.reduce(1, *)
        return values.asType(.float32).reshaped([count, elementCount])
    }

    if space is MultiBinary {
        let elementShape = space.shape ?? []
        let elementCount = elementShape.reduce(1, *)
        return values.asType(.float32).reshaped([count, elementCount])
    }

    if let md = space as? MultiDiscrete {
        let nvec = md.nvec.asType(.int32).asArray(Int32.self).map { Int($0) }
        let cols = nvec.count
        let total = nvec.reduce(0, +)

        let xs = values.asType(.int32).reshaped([count * cols]).asArray(Int32.self).map { Int($0) }
        var out = [Float](repeating: 0, count: count * total)

        for i in 0..<count {
            var outOffset = i * total
            let inOffset = i * cols
            for j in 0..<cols {
                let n = nvec[j]
                let v = xs[inOffset + j]
                if v >= 0 && v < n {
                    out[outOffset + v] = 1
                }
                outOffset += n
            }
        }

        return MLXArray(out).asType(.float32).reshaped([count, total])
    }

    let elementShape = space.shape ?? []
    let elementCount = elementShape.reduce(1, *)
    return values.asType(.float32).reshaped([count, elementCount])
}

private func unflattenGraphValues(space: any TensorSpace, values: MLXArray, count: Int) -> MLXArray
{
    if let box = space as? Box {
        let elementShape = box.shape ?? []
        return values.asType(box.dtype ?? .float32).reshaped([count] + elementShape)
    }

    if let mb = space as? MultiBinary {
        let elementShape = mb.shape ?? []
        let elementCount = elementShape.reduce(1, *)
        let vals = values.asType(.float32).reshaped([count * elementCount]).asArray(Float.self)
        let out = vals.map { $0 >= 0.5 ? Int32(1) : Int32(0) }
        return MLXArray(out).asType(.int32).reshaped([count] + elementShape)
    }

    if let md = space as? MultiDiscrete {
        let nvec = md.nvec.asType(.int32).asArray(Int32.self).map { Int($0) }
        let cols = nvec.count
        let total = nvec.reduce(0, +)

        let vals = values.asType(.float32).reshaped([count * total]).asArray(Float.self)
        var out: [Int32] = []
        out.reserveCapacity(count * cols)

        for i in 0..<count {
            var offset = i * total
            for n in nvec {
                var best = 0
                var bestV = -Float.infinity
                for j in 0..<n {
                    let v = vals[offset + j]
                    if v > bestV {
                        bestV = v
                        best = j
                    }
                }
                out.append(Int32(best))
                offset += n
            }
        }

        return MLXArray(out).asType(.int32).reshaped([count, cols])
    }

    let elementShape = space.shape ?? []
    return values.asType(.float32).reshaped([count] + elementShape)
}

private func concatenateFlat(_ arrays: [MLXArray]) -> MLXArray {
    var out: [Float] = []
    for a in arrays {
        out.append(contentsOf: a.asType(.float32).reshaped([-1]).asArray(Float.self))
    }
    return MLXArray(out).asType(.float32)
}
