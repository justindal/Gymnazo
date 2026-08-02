/// Applies a typed transform to observations and exposes an explicit output space.
public struct TransformObservationValueEnvironment<
    Base: ValueEnvironment,
    Output: Sendable,
    OutputSpace: ValueSpace
>: ValueEnvironment where OutputSpace.Value == Output {
    public typealias Observation = Output
    public typealias Action = Base.Action
    public typealias ObservationSpace = OutputSpace
    public typealias ActionSpace = Base.ActionSpace

    public var base: Base
    public let valueObservationSpace: OutputSpace
    public let transform: @Sendable (Base.Observation) throws -> Output

    public var valueActionSpace: ActionSpace {
        base.valueActionSpace
    }

    /// Creates an observation-transforming wrapper.
    ///
    /// - Parameters:
    ///   - base: Environment providing the source observations.
    ///   - observationSpace: Space for transformed observations.
    ///   - transform: Conversion applied to each observation.
    public init(
        _ base: Base,
        observationSpace: OutputSpace,
        transform: @escaping @Sendable (Base.Observation) throws -> Output
    ) {
        self.base = base
        self.valueObservationSpace = observationSpace
        self.transform = transform
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Output> {
        let result = try base.resetValue(seed: seed, options: options)
        return ValueReset(
            observation: try transform(result.observation),
            info: result.info
        )
    }

    public mutating func stepValue(_ action: Action) throws -> ValueStep<Output> {
        let result = try base.stepValue(action)
        return ValueStep(
            observation: try transform(result.observation),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func close() {
        base.close()
    }
}

/// Applies a typed transform to rewards.
public struct TransformRewardValueEnvironment<Base: ValueEnvironment>: ValueEnvironment {
    public typealias Observation = Base.Observation
    public typealias Action = Base.Action
    public typealias ObservationSpace = Base.ObservationSpace
    public typealias ActionSpace = Base.ActionSpace

    public var base: Base
    public let transform: @Sendable (Double) throws -> Double

    public var valueObservationSpace: ObservationSpace {
        base.valueObservationSpace
    }

    public var valueActionSpace: ActionSpace {
        base.valueActionSpace
    }

    /// Creates a reward-transforming wrapper.
    ///
    /// - Parameters:
    ///   - base: Environment whose rewards are transformed.
    ///   - transform: Conversion applied to each reward.
    public init(
        _ base: Base,
        transform: @escaping @Sendable (Double) throws -> Double
    ) {
        self.base = base
        self.transform = transform
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation> {
        try base.resetValue(seed: seed, options: options)
    }

    public mutating func stepValue(_ action: Action) throws -> ValueStep<Observation> {
        let result = try base.stepValue(action)
        return ValueStep(
            observation: result.observation,
            reward: try transform(result.reward),
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func close() {
        base.close()
    }
}

/// Transforms typed input actions into the wrapped environment's action type.
public struct TransformActionValueEnvironment<
    Base: ValueEnvironment,
    Input: Sendable,
    InputSpace: ValueSpace
>: ValueEnvironment where InputSpace.Value == Input {
    public typealias Observation = Base.Observation
    public typealias Action = Input
    public typealias ObservationSpace = Base.ObservationSpace
    public typealias ActionSpace = InputSpace

    public var base: Base
    public let valueActionSpace: InputSpace
    public let transform: @Sendable (Input) throws -> Base.Action

    public var valueObservationSpace: ObservationSpace {
        base.valueObservationSpace
    }

    /// Creates an action-transforming wrapper with an explicit input space.
    ///
    /// - Parameters:
    ///   - base: Environment receiving transformed actions.
    ///   - actionSpace: Space for actions accepted by this wrapper.
    ///   - transform: Conversion from wrapper actions to base actions.
    public init(
        _ base: Base,
        actionSpace: InputSpace,
        transform: @escaping @Sendable (Input) throws -> Base.Action
    ) {
        self.base = base
        self.valueActionSpace = actionSpace
        self.transform = transform
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation> {
        try base.resetValue(seed: seed, options: options)
    }

    public mutating func stepValue(_ action: Input) throws -> ValueStep<Observation> {
        try base.stepValue(transform(action))
    }

    public mutating func close() {
        base.close()
    }
}
