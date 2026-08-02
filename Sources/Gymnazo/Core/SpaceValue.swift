/// A type-erased value used at actor and registry boundaries.
public enum SpaceValue: Sendable, Equatable {
    case bool(Bool)
    case double(Double)
    case continuous([Float], shape: [Int])
    case discrete(Int)
    case multiDiscrete([Int])
    case multiBinary([Bool])
    case text(String)
    case tuple([SpaceValue])
    case dictionary([String: SpaceValue])
    case floatTensor(TensorValue<Float>)
    case intTensor(TensorValue<Int>)
    case boolTensor(TensorValue<Bool>)
    case sequence([SpaceValue])
    case graph(GraphSpaceValue)
}

/// Typed metadata returned by value environments.
public typealias ValueInfo = [String: SpaceValue]

/// A type-erased graph value containing nodes and directed edges.
public struct GraphSpaceValue: Sendable, Equatable {
    public let nodes: [SpaceValue]
    public let edges: [GraphSpaceEdgeValue]

    /// Creates a graph value from ordered nodes and edges.
    public init(nodes: [SpaceValue], edges: [GraphSpaceEdgeValue]) {
        self.nodes = nodes
        self.edges = edges
    }
}

/// A directed edge in a type-erased graph value.
public struct GraphSpaceEdgeValue: Sendable, Equatable {
    public let source: Int
    public let target: Int
    public let value: SpaceValue

    /// Creates an edge between two node indices.
    public init(source: Int, target: Int, value: SpaceValue) {
        self.source = source
        self.target = target
        self.value = value
    }
}
