public enum PolicyAction: Sendable, Equatable {
    case discrete(Int)
    case multiDiscrete([Int])
    case multiBinary([Bool])
    case continuous([Float])
}
