public struct ValueCollectionFrame<Observation: Sendable, Action: Sendable>: Sendable {
    public let observations: [Observation]
    public let actions: [Action]
    public let step: ValueVectorStep<Observation>

    public init(
        observations: [Observation],
        actions: [Action],
        step: ValueVectorStep<Observation>
    ) {
        self.observations = observations
        self.actions = actions
        self.step = step
    }
}

public struct ValueCollection<Observation: Sendable, Action: Sendable>: Sendable {
    public let reset: ValueVectorReset<Observation>
    public let frames: [ValueCollectionFrame<Observation, Action>]

    public init(
        reset: ValueVectorReset<Observation>,
        frames: [ValueCollectionFrame<Observation, Action>]
    ) {
        self.reset = reset
        self.frames = frames
    }
}

public actor ValueCollectionSession<Environment: ValueEnvironment> {
    public typealias Policy =
        @Sendable ([Environment.Observation]) async throws
        -> [Environment.Action]

    public nonisolated let numEnvs: Int

    private var environments: SyncValueVectorEnvironment<Environment>
    private var isCollecting = false

    public init(environments: consuming [Environment]) throws {
        let vector = try SyncValueVectorEnvironment(
            environments: environments,
            autoresetMode: .sameStep
        )
        self.numEnvs = vector.numEnvs
        self.environments = vector
    }

    public func collect(
        steps: Int,
        seed: UInt64? = nil,
        policy: Policy
    ) async throws -> ValueCollection<Environment.Observation, Environment.Action> {
        guard steps >= 0 else {
            throw GymnazoError.invalidConfiguration("Collection steps must be non-negative.")
        }
        guard !isCollecting else {
            throw GymnazoError.invalidState("A collection session is already running.")
        }
        isCollecting = true
        defer { isCollecting = false }

        let reset = try environments.reset(seed: seed)
        var observations = reset.observations
        var frames: [ValueCollectionFrame<Environment.Observation, Environment.Action>] = []
        frames.reserveCapacity(steps)

        for _ in 0..<steps {
            try Task.checkCancellation()
            let actions = try await policy(observations)
            try Task.checkCancellation()
            let step = try environments.step(actions)
            frames.append(
                ValueCollectionFrame(
                    observations: observations,
                    actions: actions,
                    step: step
                )
            )
            observations = step.observations
        }
        return ValueCollection(reset: reset, frames: frames)
    }

    public func close() throws {
        guard !isCollecting else {
            throw GymnazoError.invalidState("Cannot close an active collection session.")
        }
        environments.close()
    }
}
