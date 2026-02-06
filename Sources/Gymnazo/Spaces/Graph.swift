import MLX

/// A type-erased view of a ``Graph`` space.
public protocol AnyGraphSpace {
    var maxNodes: Int { get }
    var maxEdges: Int { get }
    var allowSelfLoops: Bool { get }
    var directed: Bool { get }
    var nodeSpaceAny: any TensorSpace { get }
    var edgeSpaceAny: any TensorSpace { get }
}

/// A padded, fixed-shape representation of a sampled graph.
public struct GraphSample {
    public let nodes: MLXArray
    public let edges: MLXArray
    public let edgeLinks: MLXArray
    public let nodeMask: MLXArray
    public let edgeMask: MLXArray

    /// Creates a graph sample.
    ///
    /// - Parameters:
    ///   - nodes: Node feature tensor of shape `[maxNodes] + nodeShape`.
    ///   - edges: Edge feature tensor of shape `[maxEdges] + edgeShape`.
    ///   - edgeLinks: Edge endpoints of shape `[maxEdges, 2]` with `-1` for padded edges.
    ///   - nodeMask: Boolean vector of shape `[maxNodes]`.
    ///   - edgeMask: Boolean vector of shape `[maxEdges]`.
    public init(nodes: MLXArray, edges: MLXArray, edgeLinks: MLXArray, nodeMask: MLXArray, edgeMask: MLXArray) {
        self.nodes = nodes
        self.edges = edges
        self.edgeLinks = edgeLinks
        self.nodeMask = nodeMask
        self.edgeMask = edgeMask
    }
}

/// A space representing graph-structured samples with node and edge features.
public struct Graph<NodeSpace: TensorSpace, EdgeSpace: TensorSpace>: Space, AnyGraphSpace {
    public let nodeSpace: NodeSpace
    public let edgeSpace: EdgeSpace
    public let maxNodes: Int
    public let maxEdges: Int
    public let allowSelfLoops: Bool
    public let directed: Bool

    private let nodeShape: [Int]
    private let edgeShape: [Int]
    private let nodeCount: Int
    private let edgeCount: Int

    /// Creates a graph space.
    ///
    /// - Parameters:
    ///   - nodeSpace: The node feature space.
    ///   - edgeSpace: The edge feature space.
    ///   - maxNodes: Maximum number of nodes in padded samples.
    ///   - maxEdges: Maximum number of edges in padded samples.
    ///   - allowSelfLoops: Whether sampled edges may connect a node to itself.
    ///   - directed: Whether edges are directed.
    public init(
        nodeSpace: NodeSpace,
        edgeSpace: EdgeSpace,
        maxNodes: Int,
        maxEdges: Int,
        allowSelfLoops: Bool = true,
        directed: Bool = true
    ) {
        precondition(maxNodes >= 0, "maxNodes must be non-negative")
        precondition(maxEdges >= 0, "maxEdges must be non-negative")
        guard let ns = nodeSpace.shape else {
            fatalError("Graph requires nodeSpace with a defined shape")
        }
        guard let es = edgeSpace.shape else {
            fatalError("Graph requires edgeSpace with a defined shape")
        }
        self.nodeSpace = nodeSpace
        self.edgeSpace = edgeSpace
        self.maxNodes = maxNodes
        self.maxEdges = maxEdges
        self.allowSelfLoops = allowSelfLoops
        self.directed = directed
        self.nodeShape = ns
        self.edgeShape = es
        self.nodeCount = ns.reduce(1, *)
        self.edgeCount = es.reduce(1, *)
    }

    public var shape: [Int]? { nil }
    public var dtype: DType? { nil }
    public var nodeSpaceAny: any TensorSpace { nodeSpace }
    public var edgeSpaceAny: any TensorSpace { edgeSpace }

    /// Samples a graph and returns a flattened representation.
    /// Use `sampleGraph` for the full `GraphSample` struct.
    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> MLXArray {
        let g = sampleGraph(key: key, mask: mask, probability: probability)
        return MLX.concatenated([
            g.nodes.flattened(),
            g.edges.flattened(),
            g.edgeLinks.flattened(),
            g.nodeMask.asType(.int32).flattened(),
            g.edgeMask.asType(.int32).flattened()
        ])
    }

    /// Samples a graph with random node/edge counts and returns a padded ``GraphSample``.
    ///
    /// - Note: `mask` and `probability` are currently ignored.
    public func sampleGraph(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> GraphSample {
        let keys = MLX.split(key: key, into: 4)
        let nodeCountKey = keys[0]
        let edgeCountKey = keys[1]
        let nodeKey = keys[2]
        let edgeKey = keys[3]

        let sampledNodes = maxNodes == 0 ? 0 : Int(MLX.randInt(low: 1, high: maxNodes + 1, key: nodeCountKey).item(Int32.self))
        let sampledEdges = maxEdges == 0 ? 0 : Int(MLX.randInt(low: 0, high: maxEdges + 1, key: edgeCountKey).item(Int32.self))

        let nodes = nodeSpace.sampleBatch(key: nodeKey, count: maxNodes)
        let edges = edgeSpace.sampleBatch(key: edgeKey, count: maxEdges)

        var nodeMaskI32 = [Int32](repeating: 0, count: maxNodes)
        if sampledNodes > 0 {
            for i in 0..<min(sampledNodes, maxNodes) {
                nodeMaskI32[i] = 1
            }
        }
        let nodeMaskArr = MLXArray(nodeMaskI32).asType(.bool)

        var edgeMaskI32 = [Int32](repeating: 0, count: maxEdges)
        if sampledEdges > 0 {
            for i in 0..<min(sampledEdges, maxEdges) {
                edgeMaskI32[i] = 1
            }
        }
        let edgeMaskArr = MLXArray(edgeMaskI32).asType(.bool)

        var links = [Int32](repeating: -1, count: maxEdges * 2)

        if sampledEdges > 0 && sampledNodes > 0 {
            let linkKey = MLX.split(key: nodeKey, into: 2)[1]
            let rand = MLX.uniform(low: 0, high: 1, [sampledEdges, 2], key: linkKey).asType(.float32)
            let idx = (rand * Float(sampledNodes)).asType(.int32).asArray(Int32.self)

            for e in 0..<sampledEdges {
                let src = idx[e * 2]
                var dst = idx[e * 2 + 1]
                if !allowSelfLoops && sampledNodes > 1 && src == dst {
                    dst = (dst + 1) % Int32(sampledNodes)
                }
                links[e * 2] = src
                links[e * 2 + 1] = dst
            }
        }

        let edgeLinks = MLXArray(links).reshaped([maxEdges, 2]).asType(.int32)

        return GraphSample(
            nodes: nodes,
            edges: edges,
            edgeLinks: edgeLinks,
            nodeMask: nodeMaskArr,
            edgeMask: edgeMaskArr
        )
    }

    /// Returns `true` if the array matches expected flattened size.
    public func contains(_ x: MLXArray) -> Bool {
        let expectedSize = maxNodes * nodeCount + maxEdges * edgeCount + maxEdges * 2 + maxNodes + maxEdges
        return x.size == expectedSize
    }

    public func containsGraph(_ x: GraphSample) -> Bool {
        if x.nodeMask.shape != [maxNodes] { return false }
        if x.edgeMask.shape != [maxEdges] { return false }
        if x.edgeLinks.shape != [maxEdges, 2] { return false }

        if x.nodes.shape.count < 1 { return false }
        if x.nodes.shape[0] != maxNodes { return false }
        if Array(x.nodes.shape.dropFirst()) != nodeShape { return false }

        if x.edges.shape.count < 1 { return false }
        if x.edges.shape[0] != maxEdges { return false }
        if Array(x.edges.shape.dropFirst()) != edgeShape { return false }

        let nodeMaskVals = x.nodeMask.asType(.bool).asArray(Bool.self)
        var nodePrefix = 0
        while nodePrefix < nodeMaskVals.count && nodeMaskVals[nodePrefix] {
            nodePrefix += 1
        }
        for i in nodePrefix..<nodeMaskVals.count {
            if nodeMaskVals[i] { return false }
        }

        let edgeMaskVals = x.edgeMask.asType(.bool).asArray(Bool.self)
        var edgePrefix = 0
        while edgePrefix < edgeMaskVals.count && edgeMaskVals[edgePrefix] {
            edgePrefix += 1
        }
        for i in edgePrefix..<edgeMaskVals.count {
            if edgeMaskVals[i] { return false }
        }

        let links = x.edgeLinks.asType(.int32).reshaped([maxEdges * 2]).asArray(Int32.self)
        for e in 0..<edgePrefix {
            let src = links[e * 2]
            let dst = links[e * 2 + 1]
            if src < 0 || dst < 0 { return false }
            if Int(src) >= nodePrefix || Int(dst) >= nodePrefix { return false }
            if !allowSelfLoops && src == dst { return false }
        }
        for e in edgePrefix..<maxEdges {
            let src = links[e * 2]
            let dst = links[e * 2 + 1]
            if src != -1 || dst != -1 { return false }
        }

        let nd = nodeSpace.dtype ?? .float32
        if nd != .float32 && nd != .int32 {
            return false
        }

        let ed = edgeSpace.dtype ?? .float32
        if ed != .float32 && ed != .int32 {
            return false
        }

        let flatNodes = x.nodes.asType(nd).reshaped([maxNodes, nodeCount])
        if nd == .float32 {
            let allVals = flatNodes.asArray(Float.self)
            for i in 0..<nodePrefix {
                let start = i * nodeCount
                let end = start + nodeCount
                let row = MLXArray(Array(allVals[start..<end])).reshaped(nodeShape).asType(.float32)
                if !nodeSpace.contains(row) { return false }
            }
        } else {
            let allVals = flatNodes.asArray(Int32.self)
            for i in 0..<nodePrefix {
                let start = i * nodeCount
                let end = start + nodeCount
                let row = MLXArray(Array(allVals[start..<end])).reshaped(nodeShape).asType(.int32)
                if !nodeSpace.contains(row) { return false }
            }
        }

        let flatEdges = x.edges.asType(ed).reshaped([maxEdges, edgeCount])
        if ed == .float32 {
            let allVals = flatEdges.asArray(Float.self)
            for i in 0..<edgePrefix {
                let start = i * edgeCount
                let end = start + edgeCount
                let row = MLXArray(Array(allVals[start..<end])).reshaped(edgeShape).asType(.float32)
                if !edgeSpace.contains(row) { return false }
            }
        } else {
            let allVals = flatEdges.asArray(Int32.self)
            for i in 0..<edgePrefix {
                let start = i * edgeCount
                let end = start + edgeCount
                let row = MLXArray(Array(allVals[start..<end])).reshaped(edgeShape).asType(.int32)
                if !edgeSpace.contains(row) { return false }
            }
        }

        return true
    }
}
