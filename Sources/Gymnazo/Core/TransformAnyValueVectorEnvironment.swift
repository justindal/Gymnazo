/// Applies typed observation, action, and reward transforms to a vector actor.
public actor TransformAnyValueVectorEnvironment {
    public nonisolated let numEnvs: Int
    public nonisolated let observationSpace: AnyValueSpace
    public nonisolated let actionSpace: AnyValueSpace

    private let base: AnyValueVectorEnvironment
    private let transformObservations: @Sendable ([SpaceValue]) throws -> [SpaceValue]
    private let transformActions: @Sendable ([SpaceValue]) throws -> [SpaceValue]
    private let transformRewards: @Sendable ([Double]) throws -> [Double]
    private var isPerforming = false

    /// Wraps a vector with explicit transformed space contracts.
    ///
    /// - Parameters:
    ///   - base: The vector to wrap.
    ///   - observationSpace: Optional transformed observation space.
    ///   - actionSpace: Optional transformed action space.
    ///   - transformObservations: Batch observation transform.
    ///   - transformActions: Batch action transform applied before stepping.
    ///   - transformRewards: Batch reward transform.
    public init(
        _ base: AnyValueVectorEnvironment,
        observationSpace: AnyValueSpace? = nil,
        actionSpace: AnyValueSpace? = nil,
        transformObservations: @escaping @Sendable ([SpaceValue]) throws -> [SpaceValue] = { $0 },
        transformActions: @escaping @Sendable ([SpaceValue]) throws -> [SpaceValue] = { $0 },
        transformRewards: @escaping @Sendable ([Double]) throws -> [Double] = { $0 }
    ) {
        self.base = base
        self.numEnvs = base.numEnvs
        self.observationSpace = observationSpace ?? base.observationSpace
        self.actionSpace = actionSpace ?? base.actionSpace
        self.transformObservations = transformObservations
        self.transformActions = transformActions
        self.transformRewards = transformRewards
    }

    /// Resets the wrapped vector and transforms its observations.
    ///
    /// - Parameters:
    ///   - seed: Optional base seed.
    ///   - options: Reset options passed to the base vector.
    /// - Returns: The transformed reset result.
    /// - Throws: If the base operation or observation transform fails.
    public func reset(
        seed: UInt64? = nil,
        options: ValueResetOptions? = nil
    ) async throws -> ValueVectorReset<SpaceValue> {
        try begin()
        defer { isPerforming = false }
        let result = try await base.reset(seed: seed, options: options)
        let observations = try checkedObservations(
            transformObservations(result.observations)
        )
        return ValueVectorReset(observations: observations, infos: result.infos)
    }

    /// Transforms actions, steps the vector, and transforms its outputs.
    ///
    /// - Parameter actions: One action for each environment slot.
    /// - Returns: The transformed step result.
    /// - Throws: If a transform returns an invalid batch or the base operation fails.
    public func step(
        _ actions: [SpaceValue]
    ) async throws -> ValueVectorStep<SpaceValue> {
        try begin()
        defer { isPerforming = false }
        let transformedActions = try transformActions(actions)
        guard transformedActions.count == numEnvs,
            transformedActions.allSatisfy(base.actionSpace.containsValue)
        else {
            throw GymnazoError.invalidState(
                "A vector action transform returned values outside the base space."
            )
        }
        let result = try await base.step(transformedActions)
        let observations = try checkedObservations(
            transformObservations(result.observations)
        )
        let rewards = try transformRewards(result.rewards)
        guard rewards.count == numEnvs, rewards.allSatisfy(\.isFinite) else {
            throw GymnazoError.invalidState(
                "A vector reward transform returned an invalid batch."
            )
        }
        var finalObservations = result.finalObservations
        let finalIndices = finalObservations.indices.filter {
            finalObservations[$0] != nil
        }
        if !finalIndices.isEmpty {
            let values = finalIndices.compactMap { finalObservations[$0] }
            let transformed = try transformObservations(values)
            guard transformed.count == values.count,
                transformed.allSatisfy(observationSpace.containsValue)
            else {
                throw GymnazoError.invalidState(
                    "A vector observation transform returned invalid terminal values."
                )
            }
            for (index, value) in zip(finalIndices, transformed) {
                finalObservations[index] = value
            }
        }
        return ValueVectorStep(
            observations: observations,
            rewards: rewards,
            terminations: result.terminations,
            truncations: result.truncations,
            infos: result.infos,
            finalObservations: finalObservations,
            finalInfos: result.finalInfos,
            resetInfos: result.resetInfos
        )
    }

    /// Closes the wrapped vector.
    public func close() async throws {
        try begin()
        defer { isPerforming = false }
        try await base.close()
    }

    private func begin() throws {
        guard !isPerforming else {
            throw GymnazoError.invalidState("A vector wrapper operation is already active.")
        }
        isPerforming = true
    }

    private func checkedObservations(
        _ observations: [SpaceValue]
    ) throws -> [SpaceValue] {
        guard observations.count == numEnvs,
            observations.allSatisfy(observationSpace.containsValue)
        else {
            throw GymnazoError.invalidState(
                "A vector observation transform returned an invalid batch."
            )
        }
        return observations
    }
}
