protocol AnyValueEnvironmentStorage: AnyObject {
    func reset(
        seed: UInt64?,
        options: ValueResetOptions?,
        isolation: isolated (any Actor)?
    ) async throws -> ValueReset<SpaceValue>
    func step(
        _ action: SpaceValue,
        isolation: isolated (any Actor)?
    ) async throws -> ValueStep<SpaceValue>
    func close()
}

private final class ValueEnvironmentStorage<Base: ValueEnvironment>: AnyValueEnvironmentStorage
where Base.ObservationSpace: SpaceValueConvertible, Base.ActionSpace: SpaceValueConvertible {
    private var base: Base

    init(_ base: consuming Base) {
        self.base = base
    }

    func reset(
        seed: UInt64?,
        options: ValueResetOptions?,
        isolation: isolated (any Actor)?
    ) async throws -> ValueReset<SpaceValue> {
        let result = try base.resetValue(seed: seed, options: options)
        return ValueReset(
            observation: base.valueObservationSpace.encodeValue(result.observation),
            info: result.info
        )
    }

    func step(
        _ action: SpaceValue,
        isolation: isolated (any Actor)?
    ) async throws -> ValueStep<SpaceValue> {
        guard let decoded = base.valueActionSpace.decodeValue(action) else {
            throw GymnazoError.invalidActionType(
                expected: String(describing: Base.Action.self),
                actual: String(describing: action)
            )
        }
        let result = try base.stepValue(decoded)
        return ValueStep(
            observation: base.valueObservationSpace.encodeValue(result.observation),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    func close() {
        base.close()
    }
}

public actor AnyValueEnvironment {
    public nonisolated let id: String?
    public nonisolated let observationSpace: AnyValueSpace
    public nonisolated let actionSpace: AnyValueSpace

    private let storage: any AnyValueEnvironmentStorage
    private var state: State = .initial
    private var maxEpisodeSteps: Int?
    private var elapsedSteps = 0

    public init<Base: ValueEnvironment>(id: String? = nil, _ base: consuming Base)
    where Base.ObservationSpace: SpaceValueConvertible, Base.ActionSpace: SpaceValueConvertible {
        self.id = id
        self.observationSpace = AnyValueSpace(base.valueObservationSpace)
        self.actionSpace = AnyValueSpace(base.valueActionSpace)
        self.storage = ValueEnvironmentStorage(base)
    }

    init(
        id: String,
        storage: sending any AnyValueEnvironmentStorage,
        observationSpace: AnyValueSpace,
        actionSpace: AnyValueSpace
    ) {
        self.id = id
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace
        self.storage = storage
    }

    public func reset(
        seed: UInt64? = nil,
        options: ValueResetOptions? = nil
    ) async throws -> ValueReset<SpaceValue> {
        guard state != .closed else {
            throw GymnazoError.invalidState("The environment is closed.")
        }
        let result = try await storage.reset(
            seed: seed,
            options: options,
            isolation: self
        )
        guard observationSpace.containsValue(result.observation) else {
            throw GymnazoError.observationOutsideSpace(envId: id)
        }
        elapsedSteps = 0
        state = .active
        return result
    }

    public func step(_ action: SpaceValue) async throws -> ValueStep<SpaceValue> {
        switch state {
        case .initial:
            throw GymnazoError.stepBeforeReset
        case .finished:
            throw GymnazoError.invalidState(
                "Reset the environment before stepping after an episode ends.")
        case .closed:
            throw GymnazoError.invalidState("The environment is closed.")
        case .active:
            break
        }
        guard actionSpace.containsValue(action) else {
            throw GymnazoError.actionOutsideSpace(envId: id)
        }
        var result = try await storage.step(action, isolation: self)
        guard observationSpace.containsValue(result.observation) else {
            throw GymnazoError.observationOutsideSpace(envId: id)
        }
        elapsedSteps += 1
        if let maxEpisodeSteps,
            elapsedSteps >= maxEpisodeSteps,
            !result.truncated
        {
            var info = result.info
            info[EnvInfoKey.timeLimitTruncated] = .bool(!result.terminated)
            result = ValueStep(
                observation: result.observation,
                reward: result.reward,
                terminated: result.terminated,
                truncated: true,
                info: info
            )
        }
        if result.terminated || result.truncated {
            state = .finished
        }
        return result
    }

    func applyTimeLimit(_ maxEpisodeSteps: Int?) throws {
        guard state == .initial else {
            throw GymnazoError.invalidState(
                "Configure the time limit before resetting the environment.")
        }
        if let maxEpisodeSteps, maxEpisodeSteps <= 0 {
            throw GymnazoError.invalidMaxEpisodeSteps(maxEpisodeSteps)
        }
        self.maxEpisodeSteps = maxEpisodeSteps
    }

    public func close() {
        guard state != .closed else { return }
        storage.close()
        state = .closed
    }

    private enum State: Equatable {
        case initial
        case active
        case finished
        case closed
    }
}
