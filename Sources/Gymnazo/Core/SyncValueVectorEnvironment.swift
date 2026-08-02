public struct ValueVectorReset<Observation: Sendable>: Sendable {
    public let observations: [Observation]
    public let infos: [ValueInfo]

    public init(observations: [Observation], infos: [ValueInfo]) {
        self.observations = observations
        self.infos = infos
    }
}

public struct ValueVectorStep<Observation: Sendable>: Sendable {
    public let observations: [Observation]
    public let rewards: [Double]
    public let terminations: [Bool]
    public let truncations: [Bool]
    public let infos: [ValueInfo]
    public let finalObservations: [Observation?]
    public let finalInfos: [ValueInfo?]
    public let resetInfos: [ValueInfo?]

    public init(
        observations: [Observation],
        rewards: [Double],
        terminations: [Bool],
        truncations: [Bool],
        infos: [ValueInfo],
        finalObservations: [Observation?],
        finalInfos: [ValueInfo?],
        resetInfos: [ValueInfo?]
    ) {
        self.observations = observations
        self.rewards = rewards
        self.terminations = terminations
        self.truncations = truncations
        self.infos = infos
        self.finalObservations = finalObservations
        self.finalInfos = finalInfos
        self.resetInfos = resetInfos
    }
}

public struct SyncValueVectorEnvironment<Environment: ValueEnvironment> {
    public let numEnvs: Int
    public let autoresetMode: AutoresetMode
    public private(set) var isClosed = false

    private var environments: [Environment]
    private var needsReset: [Bool]
    private var hasReset = false
    private var isInvalid = false

    public init(
        environments: [Environment],
        autoresetMode: AutoresetMode = .nextStep
    ) throws {
        guard !environments.isEmpty else {
            throw GymnazoError.invalidNumEnvs(environments.count)
        }
        self.numEnvs = environments.count
        self.environments = environments
        self.autoresetMode = autoresetMode
        self.needsReset = Array(repeating: true, count: environments.count)
    }

    public mutating func reset(
        seed: UInt64? = nil,
        options: ValueResetOptions? = nil
    ) throws -> ValueVectorReset<Environment.Observation> {
        try ensureOpen()

        do {
            var observations: [Environment.Observation] = []
            var infos: [ValueInfo] = []
            observations.reserveCapacity(numEnvs)
            infos.reserveCapacity(numEnvs)

            for index in environments.indices {
                let environmentSeed = try derivedSeed(for: index, base: seed)
                let result = try environments[index].resetValue(
                    seed: environmentSeed, options: options)
                observations.append(result.observation)
                infos.append(result.info)
                needsReset[index] = false
            }

            isInvalid = false
            hasReset = true
            return ValueVectorReset(observations: observations, infos: infos)
        } catch {
            isInvalid = true
            throw error
        }
    }

    public mutating func step(_ actions: [Environment.Action]) throws -> ValueVectorStep<
        Environment.Observation
    > {
        try ensureOpen()
        guard hasReset else {
            throw GymnazoError.stepBeforeReset
        }
        guard !isInvalid else {
            throw GymnazoError.invalidState(
                "Reset every vector environment after a failed vector operation.")
        }
        guard actions.count == numEnvs else {
            throw GymnazoError.vectorEnvActionCountMismatch(
                expected: numEnvs, actual: actions.count)
        }

        do {
            var observations: [Environment.Observation] = []
            var rewards: [Double] = []
            var terminations: [Bool] = []
            var truncations: [Bool] = []
            var infos: [ValueInfo] = []
            var finalObservations: [Environment.Observation?] = []
            var finalInfos: [ValueInfo?] = []
            var resetInfos: [ValueInfo?] = []

            observations.reserveCapacity(numEnvs)
            rewards.reserveCapacity(numEnvs)
            terminations.reserveCapacity(numEnvs)
            truncations.reserveCapacity(numEnvs)
            infos.reserveCapacity(numEnvs)
            finalObservations.reserveCapacity(numEnvs)
            finalInfos.reserveCapacity(numEnvs)
            resetInfos.reserveCapacity(numEnvs)

            for index in environments.indices {
                if needsReset[index] {
                    guard autoresetMode == .nextStep else {
                        throw GymnazoError.vectorEnvNeedsReset(index: index)
                    }
                    let reset = try environments[index].resetValue(seed: nil, options: nil)
                    needsReset[index] = false
                    observations.append(reset.observation)
                    rewards.append(0)
                    terminations.append(false)
                    truncations.append(false)
                    infos.append(reset.info)
                    finalObservations.append(nil)
                    finalInfos.append(nil)
                    resetInfos.append(reset.info)
                    continue
                }

                let result = try environments[index].stepValue(actions[index])
                let finished = result.terminated || result.truncated

                rewards.append(result.reward)
                terminations.append(result.terminated)
                truncations.append(result.truncated)
                infos.append(result.info)

                guard finished else {
                    observations.append(result.observation)
                    finalObservations.append(nil)
                    finalInfos.append(nil)
                    resetInfos.append(nil)
                    continue
                }

                finalObservations.append(result.observation)
                finalInfos.append(result.info)

                switch autoresetMode {
                case .sameStep:
                    let reset = try environments[index].resetValue(seed: nil, options: nil)
                    observations.append(reset.observation)
                    resetInfos.append(reset.info)
                case .nextStep:
                    needsReset[index] = true
                    observations.append(result.observation)
                    resetInfos.append(nil)
                case .disabled:
                    needsReset[index] = true
                    observations.append(result.observation)
                    resetInfos.append(nil)
                }
            }

            return ValueVectorStep(
                observations: observations,
                rewards: rewards,
                terminations: terminations,
                truncations: truncations,
                infos: infos,
                finalObservations: finalObservations,
                finalInfos: finalInfos,
                resetInfos: resetInfos
            )
        } catch {
            isInvalid = true
            throw error
        }
    }

    public mutating func close() {
        guard !isClosed else { return }
        for index in environments.indices {
            environments[index].close()
        }
        isClosed = true
    }

    private func derivedSeed(for index: Int, base: UInt64?) throws -> UInt64? {
        guard let base else { return nil }
        let result = base.addingReportingOverflow(UInt64(index))
        guard !result.overflow else {
            throw GymnazoError.invalidConfiguration("Vector environment seed overflowed UInt64.")
        }
        return result.partialValue
    }

    private func ensureOpen() throws {
        guard !isClosed else {
            throw GymnazoError.vectorEnvClosed
        }
    }
}
