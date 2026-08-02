/// Selects sequential or actor-owned parallel typed vector collection.
public enum ValueVectorMode: String, Sendable, Codable {
    case sync
    case parallel
}

private enum AnyValueVectorEvent: Sendable {
    case reset(Int, ValueReset<SpaceValue>)
    case step(Int, ValueStep<SpaceValue>)
}

/// An actor-owned vector of type-erased value environments.
///
/// Each slot owns its own environment actor. The vector preserves terminal
/// observations and reset metadata independently from current observations.
public actor AnyValueVectorEnvironment {
    public nonisolated let numEnvs: Int
    public nonisolated let autoresetMode: AutoresetMode
    public nonisolated let mode: ValueVectorMode
    public nonisolated let observationSpace: AnyValueSpace
    public nonisolated let actionSpace: AnyValueSpace

    private let environments: [AnyValueEnvironment]
    private var needsReset: [Bool]
    private var hasReset = false
    private var state: State = .ready

    init(
        environments: [AnyValueEnvironment],
        mode: ValueVectorMode,
        autoresetMode: AutoresetMode
    ) throws {
        guard let first = environments.first else {
            throw GymnazoError.invalidNumEnvs(0)
        }
        self.environments = environments
        self.numEnvs = environments.count
        self.mode = mode
        self.autoresetMode = autoresetMode
        self.observationSpace = first.observationSpace
        self.actionSpace = first.actionSpace
        self.needsReset = Array(repeating: true, count: environments.count)
    }

    /// Resets every slot using deterministic slot-derived seeds.
    ///
    /// - Parameters:
    ///   - seed: Optional base seed; each slot receives a derived seed.
    ///   - options: Reset options passed to every slot.
    /// - Returns: Batched observations and reset metadata.
    /// - Throws: If a slot reset fails or the vector is in an invalid state.
    public func reset(
        seed: UInt64? = nil,
        options: ValueResetOptions? = nil
    ) async throws -> ValueVectorReset<SpaceValue> {
        try begin(allowInvalid: true)
        do {
            let results = try await resetAll(seed: seed, options: options)
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

    /// Steps every slot with one typed action per environment.
    ///
    /// - Parameter actions: One action for each environment slot.
    /// - Returns: Batched observations, rewards, terminal flags, and metadata.
    /// - Throws: If the vector requires reset, action count is wrong, or a slot fails.
    public func step(
        _ actions: [SpaceValue]
    ) async throws -> ValueVectorStep<SpaceValue> {
        try begin(allowInvalid: false)
        guard hasReset else {
            state = .ready
            throw GymnazoError.stepBeforeReset
        }
        guard actions.count == numEnvs else {
            state = .ready
            throw GymnazoError.vectorEnvActionCountMismatch(
                expected: numEnvs,
                actual: actions.count
            )
        }

        do {
            let events = try await perform(actions)
            var observations = [SpaceValue?](repeating: nil, count: numEnvs)
            var rewards = Array(repeating: 0.0, count: numEnvs)
            var terminations = Array(repeating: false, count: numEnvs)
            var truncations = Array(repeating: false, count: numEnvs)
            var infos = [ValueInfo?](repeating: nil, count: numEnvs)
            var finalObservations = [SpaceValue?](repeating: nil, count: numEnvs)
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
                    if result.terminated || result.truncated {
                        finalObservations[index] = result.observation
                        finalInfos[index] = result.info
                        if autoresetMode == .sameStep {
                            sameStepIndices.append(index)
                        } else {
                            needsReset[index] = true
                            observations[index] = result.observation
                        }
                    } else {
                        observations[index] = result.observation
                    }
                }
            }

            let resets = try await reset(indices: sameStepIndices)
            for (index, result) in resets {
                observations[index] = result.observation
                resetInfos[index] = result.info
                needsReset[index] = false
            }

            state = .ready
            return ValueVectorStep(
                observations: try orderedValues(observations),
                rewards: rewards,
                terminations: terminations,
                truncations: truncations,
                infos: try orderedValues(infos),
                finalObservations: finalObservations,
                finalInfos: finalInfos,
                resetInfos: resetInfos
            )
        } catch {
            state = .invalid
            throw error
        }
    }

    /// Closes every slot after any active operation has completed.
    public func close() async throws {
        switch state {
        case .closed:
            return
        case .ready, .invalid:
            state = .closing
        case .performing, .closing:
            throw GymnazoError.invalidState(
                "Cannot close a vector environment during an active operation."
            )
        }
        for environment in environments {
            await environment.close()
        }
        state = .closed
    }

    private func resetAll(
        seed: UInt64?,
        options: ValueResetOptions?
    ) async throws -> [ValueReset<SpaceValue>] {
        switch mode {
        case .sync:
            var results: [ValueReset<SpaceValue>] = []
            results.reserveCapacity(numEnvs)
            for index in environments.indices {
                results.append(
                    try await environments[index].reset(
                        seed: try derivedSeed(index: index, base: seed),
                        options: options
                    )
                )
            }
            return results
        case .parallel:
            return try await withThrowingTaskGroup(
                of: (Int, ValueReset<SpaceValue>).self
            ) { group in
                for index in environments.indices {
                    let environment = environments[index]
                    let derivedSeed = try derivedSeed(index: index, base: seed)
                    group.addTask {
                        (index, try await environment.reset(seed: derivedSeed, options: options))
                    }
                }
                var values = [ValueReset<SpaceValue>?](repeating: nil, count: numEnvs)
                for try await (index, value) in group { values[index] = value }
                return try orderedValues(values)
            }
        }
    }

    private func perform(_ actions: [SpaceValue]) async throws -> [AnyValueVectorEvent] {
        switch mode {
        case .sync:
            var events: [AnyValueVectorEvent] = []
            events.reserveCapacity(numEnvs)
            for index in environments.indices {
                if needsReset[index] {
                    guard autoresetMode == .nextStep else {
                        throw GymnazoError.vectorEnvNeedsReset(index: index)
                    }
                    events.append(
                        .reset(index, try await environments[index].reset())
                    )
                } else {
                    events.append(
                        .step(index, try await environments[index].step(actions[index]))
                    )
                }
            }
            return events
        case .parallel:
            return try await withThrowingTaskGroup(of: AnyValueVectorEvent.self) { group in
                for index in environments.indices {
                    let environment = environments[index]
                    if needsReset[index] {
                        guard autoresetMode == .nextStep else {
                            throw GymnazoError.vectorEnvNeedsReset(index: index)
                        }
                        group.addTask { .reset(index, try await environment.reset()) }
                    } else {
                        let action = actions[index]
                        group.addTask { .step(index, try await environment.step(action)) }
                    }
                }
                var values = [AnyValueVectorEvent?](repeating: nil, count: numEnvs)
                for try await event in group {
                    switch event {
                    case .reset(let index, _), .step(let index, _):
                        values[index] = event
                    }
                }
                return try orderedValues(values)
            }
        }
    }

    private func reset(
        indices: [Int]
    ) async throws -> [(Int, ValueReset<SpaceValue>)] {
        guard !indices.isEmpty else { return [] }
        switch mode {
        case .sync:
            var results: [(Int, ValueReset<SpaceValue>)] = []
            for index in indices {
                results.append((index, try await environments[index].reset()))
            }
            return results
        case .parallel:
            return try await withThrowingTaskGroup(
                of: (Int, ValueReset<SpaceValue>).self
            ) { group in
                for index in indices {
                    let environment = environments[index]
                    group.addTask { (index, try await environment.reset()) }
                }
                var results: [(Int, ValueReset<SpaceValue>)] = []
                for try await result in group { results.append(result) }
                return results
            }
        }
    }

    private func begin(allowInvalid: Bool) throws {
        switch state {
        case .ready:
            state = .performing
        case .invalid where allowInvalid:
            state = .performing
        case .closed:
            throw GymnazoError.vectorEnvClosed
        case .performing, .closing:
            throw GymnazoError.invalidState("A vector environment operation is already active.")
        case .invalid:
            throw GymnazoError.invalidState(
                "Reset every vector environment after a failed vector operation."
            )
        }
    }

    private func derivedSeed(index: Int, base: UInt64?) throws -> UInt64? {
        guard let base else { return nil }
        let result = base.addingReportingOverflow(UInt64(index))
        guard !result.overflow else {
            throw GymnazoError.invalidConfiguration("Vector environment seed overflowed UInt64.")
        }
        return result.partialValue
    }

    private enum State {
        case ready
        case performing
        case invalid
        case closing
        case closed
    }
}

private func orderedValues<Value>(_ values: [Value?]) throws -> [Value] {
    var result: [Value] = []
    result.reserveCapacity(values.count)
    for value in values {
        guard let value else {
            throw GymnazoError.invalidState(
                "A vector operation completed without every result."
            )
        }
        result.append(value)
    }
    return result
}
