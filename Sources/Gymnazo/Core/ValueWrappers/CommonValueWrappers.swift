public typealias DtypeObservationValueEnvironment<
    Base: ValueEnvironment,
    Output: Sendable,
    OutputSpace: ValueSpace
> = TransformObservationValueEnvironment<Base, Output, OutputSpace>
where OutputSpace.Value == Output

public typealias ClipRewardValueEnvironment<Base: ValueEnvironment> =
    TransformRewardValueEnvironment<Base>

public struct ReshapeObservationValueEnvironment<Base: ValueEnvironment>: ValueEnvironment
where Base.Observation == TensorValue<Float>, Base.ObservationSpace == BoxValueSpace {
    public typealias Observation = TensorValue<Float>
    public typealias Action = Base.Action
    public typealias ObservationSpace = BoxValueSpace
    public typealias ActionSpace = Base.ActionSpace

    public var base: Base
    public let valueObservationSpace: BoxValueSpace

    public var valueActionSpace: ActionSpace { base.valueActionSpace }

    public init(_ base: Base, shape: [Int]) throws {
        let count = try TensorValue<Float>.elementCount(for: shape)
        guard count == base.valueObservationSpace.low.count else {
            throw GymnazoError.invalidConfiguration(
                "Reshaped observations must preserve the element count."
            )
        }
        self.base = base
        self.valueObservationSpace = try BoxValueSpace(
            low: TensorValue(base.valueObservationSpace.low.elements, shape: shape),
            high: TensorValue(base.valueObservationSpace.high.elements, shape: shape)
        )
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation> {
        let result = try base.resetValue(seed: seed, options: options)
        return ValueReset(
            observation: try reshape(result.observation),
            info: result.info
        )
    }

    public mutating func stepValue(_ action: Action) throws -> ValueStep<Observation> {
        let result = try base.stepValue(action)
        return ValueStep(
            observation: try reshape(result.observation),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func close() { base.close() }

    private func reshape(_ value: TensorValue<Float>) throws -> TensorValue<Float> {
        try TensorValue(value.elements, shape: valueObservationSpace.shape)
    }
}

public struct DelayObservationValueEnvironment<Base: ValueEnvironment>: ValueEnvironment
where Base.Observation == TensorValue<Float>, Base.ObservationSpace == BoxValueSpace {
    public typealias Observation = TensorValue<Float>
    public typealias Action = Base.Action
    public typealias ObservationSpace = BoxValueSpace
    public typealias ActionSpace = Base.ActionSpace

    public var base: Base
    public let delay: Int
    private var observations: [Observation] = []

    public var valueObservationSpace: ObservationSpace { base.valueObservationSpace }
    public var valueActionSpace: ActionSpace { base.valueActionSpace }

    public init(_ base: Base, delay: Int) throws {
        guard delay >= 0 else {
            throw GymnazoError.invalidConfiguration("Observation delay must be non-negative.")
        }
        self.base = base
        self.delay = delay
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation> {
        let result = try base.resetValue(seed: seed, options: options)
        observations = try (0..<delay).map { _ in try zero() }
        return ValueReset(observation: delayed(result.observation), info: result.info)
    }

    public mutating func stepValue(_ action: Action) throws -> ValueStep<Observation> {
        let result = try base.stepValue(action)
        return ValueStep(
            observation: delayed(result.observation),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func close() {
        observations.removeAll(keepingCapacity: false)
        base.close()
    }

    private mutating func delayed(_ observation: Observation) -> Observation {
        guard delay > 0 else { return observation }
        observations.append(observation)
        return observations.removeFirst()
    }

    private func zero() throws -> Observation {
        try TensorValue(
            [Float](repeating: 0, count: valueObservationSpace.low.count),
            shape: valueObservationSpace.shape
        )
    }
}

public struct TimeAwareObservationValueEnvironment<Base: ValueEnvironment>: ValueEnvironment
where Base.Observation == TensorValue<Float>, Base.ObservationSpace == BoxValueSpace {
    public typealias Observation = TensorValue<Float>
    public typealias Action = Base.Action
    public typealias ObservationSpace = BoxValueSpace
    public typealias ActionSpace = Base.ActionSpace

    public var base: Base
    public let maxEpisodeSteps: Int
    public let normalizeTime: Bool
    public let valueObservationSpace: BoxValueSpace
    private var elapsedSteps = 0

    public var valueActionSpace: ActionSpace { base.valueActionSpace }

    public init(
        _ base: Base,
        maxEpisodeSteps: Int,
        normalizeTime: Bool = false
    ) throws {
        guard maxEpisodeSteps > 0 else {
            throw GymnazoError.invalidMaxEpisodeSteps(maxEpisodeSteps)
        }
        let lower = base.valueObservationSpace.low.elements + [0]
        let timeUpper = normalizeTime ? Float(1) : Float(maxEpisodeSteps)
        let upper = base.valueObservationSpace.high.elements + [timeUpper]
        self.base = base
        self.maxEpisodeSteps = maxEpisodeSteps
        self.normalizeTime = normalizeTime
        self.valueObservationSpace = try BoxValueSpace(
            low: TensorValue(lower, shape: [lower.count]),
            high: TensorValue(upper, shape: [upper.count])
        )
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation> {
        let result = try base.resetValue(seed: seed, options: options)
        elapsedSteps = 0
        return ValueReset(observation: try augment(result.observation), info: result.info)
    }

    public mutating func stepValue(_ action: Action) throws -> ValueStep<Observation> {
        let result = try base.stepValue(action)
        elapsedSteps += 1
        return ValueStep(
            observation: try augment(result.observation),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func close() { base.close() }

    private func augment(_ observation: Observation) throws -> Observation {
        let time =
            normalizeTime
            ? Float(elapsedSteps) / Float(maxEpisodeSteps)
            : Float(elapsedSteps)
        let values = observation.elements + [time]
        return try TensorValue(values, shape: [values.count])
    }
}

public struct MaxAndSkipObservationValueEnvironment<Base: ValueEnvironment>: ValueEnvironment
where Base.Observation == TensorValue<Float>, Base.ObservationSpace == BoxValueSpace {
    public typealias Observation = TensorValue<Float>
    public typealias Action = Base.Action
    public typealias ObservationSpace = BoxValueSpace
    public typealias ActionSpace = Base.ActionSpace

    public var base: Base
    public let skip: Int

    public var valueObservationSpace: ObservationSpace { base.valueObservationSpace }
    public var valueActionSpace: ActionSpace { base.valueActionSpace }

    public init(_ base: Base, skip: Int = 4) throws {
        guard skip > 0 else {
            throw GymnazoError.invalidConfiguration("Frame skip must be positive.")
        }
        self.base = base
        self.skip = skip
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation> {
        try base.resetValue(seed: seed, options: options)
    }

    public mutating func stepValue(_ action: Action) throws -> ValueStep<Observation> {
        var recent: [Observation] = []
        var reward = 0.0
        var final: ValueStep<Observation>?
        for _ in 0..<skip {
            let result = try base.stepValue(action)
            reward += result.reward
            recent.append(result.observation)
            if recent.count > 2 { recent.removeFirst() }
            final = result
            if result.terminated || result.truncated { break }
        }
        guard let final else {
            throw GymnazoError.invalidState("Frame skipping produced no environment step.")
        }
        let observation = try maximum(recent)
        return ValueStep(
            observation: observation,
            reward: reward,
            terminated: final.terminated,
            truncated: final.truncated,
            info: final.info
        )
    }

    public mutating func close() { base.close() }

    private func maximum(_ values: [Observation]) throws -> Observation {
        guard let first = values.first else {
            throw GymnazoError.invalidState("Cannot maximize an empty observation buffer.")
        }
        guard values.count == 2 else { return first }
        return try TensorValue(
            zip(first.elements, values[1].elements).map(Swift.max),
            shape: first.shape
        )
    }
}

public struct StickyActionValueEnvironment<Base: ValueEnvironment>: ValueEnvironment
where Base.Action: Equatable {
    public typealias Observation = Base.Observation
    public typealias Action = Base.Action
    public typealias ObservationSpace = Base.ObservationSpace
    public typealias ActionSpace = Base.ActionSpace

    public var base: Base
    public let repeatProbability: Float
    private let initialSeed: UInt64
    private var generator: SeededGenerator
    private var previousAction: Action?

    public var valueObservationSpace: ObservationSpace { base.valueObservationSpace }
    public var valueActionSpace: ActionSpace { base.valueActionSpace }

    public init(
        _ base: Base,
        repeatProbability: Float,
        seed: UInt64 = 0
    ) throws {
        guard repeatProbability >= 0, repeatProbability <= 1 else {
            throw GymnazoError.invalidConfiguration(
                "Sticky-action probability must be between zero and one."
            )
        }
        self.base = base
        self.repeatProbability = repeatProbability
        self.initialSeed = seed
        self.generator = SeededGenerator(seed: seed)
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation> {
        generator = SeededGenerator(seed: seed ?? initialSeed)
        previousAction = nil
        return try base.resetValue(seed: seed, options: options)
    }

    public mutating func stepValue(_ action: Action) throws -> ValueStep<Observation> {
        let applied: Action
        if let previousAction,
            generator.uniform(in: Float(0)...Float(1)) < repeatProbability
        {
            applied = previousAction
        } else {
            applied = action
        }
        previousAction = applied
        return try base.stepValue(applied)
    }

    public mutating func close() { base.close() }
}

public struct FilterDictionaryObservationValueEnvironment<Base: ValueEnvironment>:
    ValueEnvironment
where
    Base.Observation == [String: SpaceValue],
    Base.ObservationSpace == DictionaryValueSpace
{
    public typealias Observation = [String: SpaceValue]
    public typealias Action = Base.Action
    public typealias ObservationSpace = DictionaryValueSpace
    public typealias ActionSpace = Base.ActionSpace

    public var base: Base
    public let keys: [String]
    public let valueObservationSpace: DictionaryValueSpace

    public var valueActionSpace: ActionSpace { base.valueActionSpace }

    public init(_ base: Base, keys: [String]) throws {
        guard Set(keys).count == keys.count else {
            throw GymnazoError.invalidConfiguration("Filter keys must be unique.")
        }
        var spaces: [String: AnyValueSpace] = [:]
        for key in keys {
            guard let space = base.valueObservationSpace.spaces[key] else {
                throw GymnazoError.invalidConfiguration("Unknown dictionary filter key: \(key).")
            }
            spaces[key] = space
        }
        self.base = base
        self.keys = keys
        self.valueObservationSpace = DictionaryValueSpace(spaces)
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation> {
        let result = try base.resetValue(seed: seed, options: options)
        return ValueReset(observation: filter(result.observation), info: result.info)
    }

    public mutating func stepValue(_ action: Action) throws -> ValueStep<Observation> {
        let result = try base.stepValue(action)
        return ValueStep(
            observation: filter(result.observation),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func close() { base.close() }

    private func filter(_ observation: Observation) -> Observation {
        Dictionary(
            uniqueKeysWithValues: keys.compactMap { key in
                observation[key].map { (key, $0) }
            })
    }
}

public struct FilterTupleObservationValueEnvironment<Base: ValueEnvironment>: ValueEnvironment
where Base.Observation == [SpaceValue], Base.ObservationSpace == TupleValueSpace {
    public typealias Observation = [SpaceValue]
    public typealias Action = Base.Action
    public typealias ObservationSpace = TupleValueSpace
    public typealias ActionSpace = Base.ActionSpace

    public var base: Base
    public let indices: [Int]
    public let valueObservationSpace: TupleValueSpace

    public var valueActionSpace: ActionSpace { base.valueActionSpace }

    public init(_ base: Base, indices: [Int]) throws {
        guard Set(indices).count == indices.count,
            indices.allSatisfy(base.valueObservationSpace.spaces.indices.contains)
        else {
            throw GymnazoError.invalidConfiguration("Tuple filter indices are invalid.")
        }
        self.base = base
        self.indices = indices
        self.valueObservationSpace = TupleValueSpace(
            indices.map { base.valueObservationSpace.spaces[$0] }
        )
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation> {
        let result = try base.resetValue(seed: seed, options: options)
        return ValueReset(observation: filter(result.observation), info: result.info)
    }

    public mutating func stepValue(_ action: Action) throws -> ValueStep<Observation> {
        let result = try base.stepValue(action)
        return ValueStep(
            observation: filter(result.observation),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    public mutating func close() { base.close() }

    private func filter(_ observation: Observation) -> Observation {
        indices.map { observation[$0] }
    }
}
