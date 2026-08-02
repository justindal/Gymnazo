private enum ValueVectorEvent<Observation: Sendable>: Sendable {
    case reset(Int, ValueReset<Observation>)
    case step(Int, ValueStep<Observation>)
}

private actor ValueEnvironmentWorker<Environment: ValueEnvironment> {
    private var environment: Environment

    init(_ environment: consuming Environment) {
        self.environment = environment
    }

    func reset(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Environment.Observation> {
        try environment.resetValue(seed: seed, options: options)
    }

    func step(_ action: Environment.Action) throws -> ValueStep<Environment.Observation> {
        try environment.stepValue(action)
    }

    func close() {
        environment.close()
    }
}

public actor ParallelValueVectorEnvironment<Environment: ValueEnvironment> {
    public let numEnvs: Int
    public let autoresetMode: AutoresetMode

    private let workers: [ValueEnvironmentWorker<Environment>]
    private var needsReset: [Bool]
    private var hasReset = false
    private var state: State = .ready

    public init(
        factories: [@Sendable () throws -> Environment],
        autoresetMode: AutoresetMode = .nextStep
    ) throws {
        guard !factories.isEmpty else {
            throw GymnazoError.invalidNumEnvs(factories.count)
        }
        self.numEnvs = factories.count
        self.autoresetMode = autoresetMode
        self.workers = try factories.map { ValueEnvironmentWorker(try $0()) }
        self.needsReset = Array(repeating: true, count: factories.count)
    }

    public func reset(
        seed: UInt64? = nil,
        options: ValueResetOptions? = nil
    ) async throws -> ValueVectorReset<Environment.Observation> {
        try begin(allowInvalidState: true)

        do {
            let results = try await withThrowingTaskGroup(
                of: (Int, ValueReset<Environment.Observation>).self
            ) { group in
                for index in workers.indices {
                    let worker = workers[index]
                    let workerSeed = try derivedSeed(for: index, base: seed)
                    group.addTask {
                        (index, try await worker.reset(seed: workerSeed, options: options))
                    }
                }

                var results = [ValueReset<Environment.Observation>?](repeating: nil, count: numEnvs)
                for try await (index, result) in group {
                    results[index] = result
                }
                return try ordered(results)
            }

            needsReset = Array(repeating: false, count: numEnvs)
            hasReset = true
            state = .ready
            return ValueVectorReset(
                observations: results.map(\.observation),
                infos: results.map(\.info)
            )
        } catch {
            state = .invalid
            throw error
        }
    }

    public func step(
        _ actions: [Environment.Action]
    ) async throws -> ValueVectorStep<Environment.Observation> {
        try begin(allowInvalidState: false)
        guard hasReset else {
            state = .ready
            throw GymnazoError.stepBeforeReset
        }
        guard actions.count == numEnvs else {
            state = .ready
            throw GymnazoError.vectorEnvActionCountMismatch(
                expected: numEnvs, actual: actions.count)
        }

        do {
            let events = try await withThrowingTaskGroup(
                of: ValueVectorEvent<Environment.Observation>.self
            ) { group in
                for index in workers.indices {
                    let worker = workers[index]
                    if needsReset[index] {
                        guard autoresetMode == .nextStep else {
                            throw GymnazoError.vectorEnvNeedsReset(index: index)
                        }
                        group.addTask {
                            .reset(index, try await worker.reset(seed: nil, options: nil))
                        }
                    } else {
                        let action = actions[index]
                        group.addTask {
                            .step(index, try await worker.step(action))
                        }
                    }
                }

                var events = [ValueVectorEvent<Environment.Observation>?](
                    repeating: nil,
                    count: numEnvs
                )
                for try await event in group {
                    switch event {
                    case .reset(let index, _), .step(let index, _):
                        events[index] = event
                    }
                }
                return try ordered(events)
            }

            var observations = [Environment.Observation?](repeating: nil, count: numEnvs)
            var rewards = Array(repeating: 0.0, count: numEnvs)
            var terminations = Array(repeating: false, count: numEnvs)
            var truncations = Array(repeating: false, count: numEnvs)
            var infos = [ValueInfo?](repeating: nil, count: numEnvs)
            var finalObservations = [Environment.Observation?](repeating: nil, count: numEnvs)
            var finalInfos = [ValueInfo?](repeating: nil, count: numEnvs)
            var resetInfos = [ValueInfo?](repeating: nil, count: numEnvs)
            var sameStepIndices: [Int] = []

            for event in events {
                switch event {
                case .reset(let index, let result):
                    needsReset[index] = false
                    observations[index] = result.observation
                    infos[index] = result.info
                    resetInfos[index] = result.info
                case .step(let index, let result):
                    rewards[index] = result.reward
                    terminations[index] = result.terminated
                    truncations[index] = result.truncated
                    infos[index] = result.info

                    guard result.terminated || result.truncated else {
                        observations[index] = result.observation
                        continue
                    }

                    finalObservations[index] = result.observation
                    finalInfos[index] = result.info

                    switch autoresetMode {
                    case .sameStep:
                        sameStepIndices.append(index)
                    case .nextStep, .disabled:
                        needsReset[index] = true
                        observations[index] = result.observation
                    }
                }
            }

            if !sameStepIndices.isEmpty {
                let resets = try await withThrowingTaskGroup(
                    of: (Int, ValueReset<Environment.Observation>).self
                ) { group in
                    for index in sameStepIndices {
                        let worker = workers[index]
                        group.addTask {
                            (index, try await worker.reset(seed: nil, options: nil))
                        }
                    }

                    var resets: [Int: ValueReset<Environment.Observation>] = [:]
                    for try await (index, result) in group {
                        resets[index] = result
                    }
                    return resets
                }

                for (index, reset) in resets {
                    observations[index] = reset.observation
                    resetInfos[index] = reset.info
                }
            }

            state = .ready
            return ValueVectorStep(
                observations: try ordered(observations),
                rewards: rewards,
                terminations: terminations,
                truncations: truncations,
                infos: try ordered(infos),
                finalObservations: finalObservations,
                finalInfos: finalInfos,
                resetInfos: resetInfos
            )
        } catch {
            state = .invalid
            throw error
        }
    }

    public func close() async throws {
        switch state {
        case .closed:
            return
        case .ready, .invalid:
            state = .closing
        case .performing, .closing:
            throw GymnazoError.invalidState(
                "Cannot close a vector environment during an active operation.")
        }
        for worker in workers {
            await worker.close()
        }
        state = .closed
    }

    private func begin(allowInvalidState: Bool) throws {
        switch state {
        case .ready:
            state = .performing
        case .invalid where allowInvalidState:
            state = .performing
        case .closed:
            throw GymnazoError.vectorEnvClosed
        case .performing, .closing:
            throw GymnazoError.invalidState(
                "A vector environment operation is already in progress.")
        case .invalid:
            throw GymnazoError.invalidState(
                "Reset every vector environment after a failed vector operation.")
        }
    }

    private func derivedSeed(for index: Int, base: UInt64?) throws -> UInt64? {
        guard let base else { return nil }
        let result = base.addingReportingOverflow(UInt64(index))
        guard !result.overflow else {
            throw GymnazoError.invalidConfiguration("Vector environment seed overflowed UInt64.")
        }
        return result.partialValue
    }

    private enum State: Equatable {
        case ready
        case performing
        case invalid
        case closing
        case closed
    }
}

private func ordered<Value>(_ values: [Value?]) throws -> [Value] {
    var ordered: [Value] = []
    ordered.reserveCapacity(values.count)
    for value in values {
        guard let value else {
            throw GymnazoError.invalidState(
                "A vector operation completed without a result for every environment.")
        }
        ordered.append(value)
    }
    return ordered
}
