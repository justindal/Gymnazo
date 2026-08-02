public struct AutoResetValueEnvironment<Base: ValueEnvironment>: ValueEnvironment
where Base.ObservationSpace: SpaceValueConvertible {
    public typealias Observation = Base.Observation
    public typealias Action = Base.Action
    public typealias ObservationSpace = Base.ObservationSpace
    public typealias ActionSpace = Base.ActionSpace

    public var base: Base
    public let mode: AutoresetMode

    private var needsReset = true

    public var valueObservationSpace: ObservationSpace {
        base.valueObservationSpace
    }

    public var valueActionSpace: ActionSpace {
        base.valueActionSpace
    }

    public init(_ base: Base, mode: AutoresetMode = .nextStep) {
        self.base = base
        self.mode = mode
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation> {
        let result = try base.resetValue(seed: seed, options: options)
        needsReset = false
        return result
    }

    public mutating func stepValue(_ action: Action) throws -> ValueStep<Observation> {
        if needsReset, mode == .nextStep {
            let result = try base.resetValue(seed: nil, options: nil)
            needsReset = false
            return ValueStep(
                observation: result.observation,
                reward: 0,
                terminated: false,
                truncated: false,
                info: result.info
            )
        }

        let result = try base.stepValue(action)
        guard result.terminated || result.truncated else {
            return result
        }
        guard mode != .disabled else {
            return result
        }
        guard mode == .sameStep else {
            needsReset = true
            return result
        }

        let reset = try base.resetValue(seed: nil, options: nil)
        var info = reset.info
        info[EnvInfoKey.finalObservation] = valueObservationSpace.encodeValue(result.observation)
        info[EnvInfoKey.finalInfo] = .dictionary(result.info)
        needsReset = false
        return ValueStep(
            observation: reset.observation,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: info
        )
    }

    public mutating func close() {
        base.close()
        needsReset = true
    }
}
