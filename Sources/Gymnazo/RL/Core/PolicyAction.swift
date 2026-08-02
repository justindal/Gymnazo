/// A sendable action returned by a policy at the typed boundary.
public enum PolicyAction: Sendable, Equatable {
    case discrete(Int)
    case multiDiscrete([Int])
    case multiBinary([Bool])
    case continuous([Float])
}
