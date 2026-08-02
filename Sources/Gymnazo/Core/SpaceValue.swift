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

public typealias ValueInfo = [String: SpaceValue]

public struct GraphSpaceValue: Sendable, Equatable {
    public let nodes: [SpaceValue]
    public let edges: [GraphSpaceEdgeValue]

    public init(nodes: [SpaceValue], edges: [GraphSpaceEdgeValue]) {
        self.nodes = nodes
        self.edges = edges
    }
}

public struct GraphSpaceEdgeValue: Sendable, Equatable {
    public let source: Int
    public let target: Int
    public let value: SpaceValue

    public init(source: Int, target: Int, value: SpaceValue) {
        self.source = source
        self.target = target
        self.value = value
    }
}
