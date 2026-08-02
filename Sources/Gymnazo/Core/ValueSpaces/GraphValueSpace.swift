/// A graph value with typed nodes and directed edges.
public struct GraphValue<Node: Sendable, Edge: Sendable>: Sendable {
    public let nodes: [Node]
    public let edges: [GraphEdge<Edge>]

    /// Creates a graph from ordered nodes and edges.
    ///
    /// - Parameters:
    ///   - nodes: Node values in graph order.
    ///   - edges: Directed edges between node indices.
    public init(nodes: [Node], edges: [GraphEdge<Edge>]) {
        self.nodes = nodes
        self.edges = edges
    }
}

/// A directed edge in a typed graph value.
public struct GraphEdge<Value: Sendable>: Sendable {
    public let source: Int
    public let target: Int
    public let value: Value

    /// Creates an edge between node indices.
    ///
    /// - Parameters:
    ///   - source: Source node index.
    ///   - target: Target node index.
    ///   - value: Edge value.
    public init(source: Int, target: Int, value: Value) {
        self.source = source
        self.target = target
        self.value = value
    }
}

/// A graph space with independent node and edge spaces.
public struct GraphValueSpace<NodeSpace: ValueSpace, EdgeSpace: ValueSpace>: ValueBatchingSpace {
    public let nodeSpace: NodeSpace
    public let edgeSpace: EdgeSpace
    public let maxNodes: Int
    public let maxEdges: Int
    public let allowSelfLoops: Bool
    public let directed: Bool

    /// Creates a graph space with optional node and edge count limits.
    ///
    /// - Parameters:
    ///   - nodeSpace: Space for node values.
    ///   - edgeSpace: Space for edge values.
    ///   - maxNodes: Maximum number of nodes.
    ///   - maxEdges: Maximum number of edges.
    ///   - allowSelfLoops: Whether an edge may target its source.
    ///   - directed: Whether edges are directed.
    /// - Throws: If graph size limits are invalid.
    public init(
        nodeSpace: NodeSpace,
        edgeSpace: EdgeSpace,
        maxNodes: Int,
        maxEdges: Int,
        allowSelfLoops: Bool = true,
        directed: Bool = true
    ) throws {
        guard maxNodes >= 0, maxEdges >= 0 else {
            throw GymnazoError.invalidConfiguration("Graph size limits must be non-negative.")
        }
        guard maxNodes > 0 || maxEdges == 0 else {
            throw GymnazoError.invalidConfiguration("A graph without nodes cannot contain edges.")
        }
        self.nodeSpace = nodeSpace
        self.edgeSpace = edgeSpace
        self.maxNodes = maxNodes
        self.maxEdges = maxEdges
        self.allowSelfLoops = allowSelfLoops
        self.directed = directed
    }

    /// Returns whether graph counts, node values, and edge values are valid.
    ///
    /// - Parameter value: The graph to validate.
    /// - Returns: True when graph structure and values match the space.
    public func containsValue(_ value: GraphValue<NodeSpace.Value, EdgeSpace.Value>) -> Bool {
        guard value.nodes.count <= maxNodes, value.edges.count <= maxEdges else { return false }
        guard value.nodes.allSatisfy(nodeSpace.containsValue) else { return false }
        return value.edges.allSatisfy { edge in
            edge.source >= 0
                && edge.target >= 0
                && edge.source < value.nodes.count
                && edge.target < value.nodes.count
                && (allowSelfLoops || edge.source != edge.target)
                && edgeSpace.containsValue(edge.value)
        }
    }

    /// Samples a graph with deterministic node and edge counts.
    ///
    /// - Parameter generator: The random generator used for sampling.
    /// - Returns: A sampled graph.
    /// - Throws: If a node or edge space cannot sample.
    public func sample(using generator: inout SeededGenerator) throws
        -> GraphValue<NodeSpace.Value, EdgeSpace.Value>
    {
        let nodeCount = maxNodes == 0 ? 0 : Int.random(in: 1...maxNodes, using: &generator)
        let canConnectNodes = allowSelfLoops || nodeCount > 1
        let edgeCount =
            nodeCount == 0 || !canConnectNodes
            ? 0
            : Int.random(in: 0...maxEdges, using: &generator)
        let nodes = try (0..<nodeCount).map { _ in
            try nodeSpace.sample(using: &generator)
        }
        let edges = try (0..<edgeCount).map { _ in
            let source = Int.random(in: 0..<nodeCount, using: &generator)
            var target = Int.random(in: 0..<nodeCount, using: &generator)
            if !allowSelfLoops, nodeCount > 1, target == source {
                target = (target + 1) % nodeCount
            }
            return GraphEdge(
                source: source,
                target: target,
                value: try edgeSpace.sample(using: &generator)
            )
        }
        return GraphValue(nodes: nodes, edges: edges)
    }
}

extension GraphValueSpace: SpaceValueConvertible
where NodeSpace: SpaceValueConvertible, EdgeSpace: SpaceValueConvertible {
    /// Erases a graph as SpaceValue.
    ///
    /// - Parameter value: The graph to encode.
    /// - Returns: The type-erased graph representation.
    public func encodeValue(_ value: GraphValue<NodeSpace.Value, EdgeSpace.Value>) -> SpaceValue {
        .graph(
            GraphSpaceValue(
                nodes: value.nodes.map(nodeSpace.encodeValue),
                edges: value.edges.map { edge in
                    GraphSpaceEdgeValue(
                        source: edge.source,
                        target: edge.target,
                        value: edgeSpace.encodeValue(edge.value)
                    )
                }
            )
        )
    }

    /// Decodes a type-erased graph when its case matches.
    ///
    /// - Parameter value: The erased value to decode.
    /// - Returns: The typed graph, or nil for another value case.
    public func decodeValue(_ value: SpaceValue) -> GraphValue<NodeSpace.Value, EdgeSpace.Value>? {
        guard case .graph(let graph) = value else { return nil }
        var nodes: [NodeSpace.Value] = []
        nodes.reserveCapacity(graph.nodes.count)
        for node in graph.nodes {
            guard let decoded = nodeSpace.decodeValue(node) else { return nil }
            nodes.append(decoded)
        }

        var edges: [GraphEdge<EdgeSpace.Value>] = []
        edges.reserveCapacity(graph.edges.count)
        for edge in graph.edges {
            guard let decoded = edgeSpace.decodeValue(edge.value) else { return nil }
            edges.append(GraphEdge(source: edge.source, target: edge.target, value: decoded))
        }
        return GraphValue(nodes: nodes, edges: edges)
    }
}

extension GraphValueSpace: ValueFlatteningSpace
where NodeSpace: ValueFlatteningSpace, EdgeSpace: ValueFlatteningSpace {
    public var flatDimension: Int? {
        nil
    }

    /// Flattens node and edge values while preserving graph structure.
    ///
    /// - Parameter value: The graph to flatten.
    /// - Returns: A flattened graph representation.
    /// - Throws: If the graph is outside the space.
    public func flattenValue(
        _ value: GraphValue<NodeSpace.Value, EdgeSpace.Value>
    ) throws -> FlattenedValue {
        guard containsValue(value) else {
            throw GymnazoError.invalidState("Cannot flatten a value outside the graph space.")
        }
        return .graph(
            FlattenedGraphValue(
                nodes: try value.nodes.map(nodeSpace.flattenValue),
                edges: try value.edges.map { edge in
                    FlattenedGraphEdge(
                        source: edge.source,
                        target: edge.target,
                        value: try edgeSpace.flattenValue(edge.value)
                    )
                }
            )
        )
    }

    /// Restores a typed graph from its flattened representation.
    ///
    /// - Parameter value: The flattened graph representation.
    /// - Returns: The restored graph.
    /// - Throws: If the representation is invalid for the space.
    public func unflattenValue(
        _ value: FlattenedValue
    ) throws -> GraphValue<NodeSpace.Value, EdgeSpace.Value> {
        guard case .graph(let graph) = value else {
            throw GymnazoError.invalidState("The flattened graph structure is invalid.")
        }
        let result = GraphValue(
            nodes: try graph.nodes.map(nodeSpace.unflattenValue),
            edges: try graph.edges.map { edge in
                GraphEdge(
                    source: edge.source,
                    target: edge.target,
                    value: try edgeSpace.unflattenValue(edge.value)
                )
            }
        )
        guard containsValue(result) else {
            throw GymnazoError.invalidState("The unflattened graph value is outside the space.")
        }
        return result
    }
}
