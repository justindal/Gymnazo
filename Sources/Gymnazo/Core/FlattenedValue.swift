/// A reversible flattened representation of a structured value.
public indirect enum FlattenedValue: Sendable, Equatable {
    case vector([Float])
    case sequence([FlattenedValue])
    case tuple([FlattenedValue])
    case dictionary([String: FlattenedValue])
    case graph(FlattenedGraphValue)
}

/// A flattened graph value retaining node and edge structure.
public struct FlattenedGraphValue: Sendable, Equatable {
    public let nodes: [FlattenedValue]
    public let edges: [FlattenedGraphEdge]

    /// Creates a flattened graph from ordered nodes and edges.
    ///
    /// - Parameters:
    ///   - nodes: Flattened node values in graph order.
    ///   - edges: Flattened directed edges.
    public init(nodes: [FlattenedValue], edges: [FlattenedGraphEdge]) {
        self.nodes = nodes
        self.edges = edges
    }
}

/// A flattened graph edge with source and target indices.
public struct FlattenedGraphEdge: Sendable, Equatable {
    public let source: Int
    public let target: Int
    public let value: FlattenedValue

    /// Creates a flattened graph edge.
    ///
    /// - Parameters:
    ///   - source: The source node index.
    ///   - target: The target node index.
    ///   - value: The flattened edge value.
    public init(source: Int, target: Int, value: FlattenedValue) {
        self.source = source
        self.target = target
        self.value = value
    }
}

/// A type-erased space that supports reversible flattening.
public protocol ValueFlatteningSpace: SpaceValueConvertible {
    /// The flattened scalar dimension when the space has a fixed size.
    var flatDimension: Int? { get }
    /// Flattens a value while preserving enough structure for round trips.
    ///
    /// - Parameter value: The typed value to flatten.
    /// - Returns: The flattened representation.
    /// - Throws: If the value is outside the space.
    func flattenValue(_ value: Value) throws -> FlattenedValue
    /// Restores a value from its flattened representation.
    ///
    /// - Parameter value: The flattened representation to restore.
    /// - Returns: The typed value.
    /// - Throws: If the representation is invalid for the space.
    func unflattenValue(_ value: FlattenedValue) throws -> Value
}
