public indirect enum FlattenedValue: Sendable, Equatable {
    case vector([Float])
    case sequence([FlattenedValue])
    case tuple([FlattenedValue])
    case dictionary([String: FlattenedValue])
    case graph(FlattenedGraphValue)
}

public struct FlattenedGraphValue: Sendable, Equatable {
    public let nodes: [FlattenedValue]
    public let edges: [FlattenedGraphEdge]

    public init(nodes: [FlattenedValue], edges: [FlattenedGraphEdge]) {
        self.nodes = nodes
        self.edges = edges
    }
}

public struct FlattenedGraphEdge: Sendable, Equatable {
    public let source: Int
    public let target: Int
    public let value: FlattenedValue

    public init(source: Int, target: Int, value: FlattenedValue) {
        self.source = source
        self.target = target
        self.value = value
    }
}

public protocol ValueFlatteningSpace: SpaceValueConvertible {
    var flatDimension: Int? { get }
    func flattenValue(_ value: Value) throws -> FlattenedValue
    func unflattenValue(_ value: FlattenedValue) throws -> Value
}
