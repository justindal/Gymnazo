extension DQN {
    /// Selects an action from a typed tensor observation.
    ///
    /// - Parameters:
    ///   - observation: The typed float observation.
    ///   - deterministic: Whether to disable policy sampling.
    /// - Returns: The selected typed policy action.
    public func act(
        observation: TensorValue<Float>,
        deterministic: Bool = true
    ) -> PolicyAction {
        act(observation: observation.elements, deterministic: deterministic)
    }
}

extension PPO {
    /// Selects an action from a typed tensor observation.
    ///
    /// - Parameters:
    ///   - observation: The typed float observation.
    ///   - deterministic: Whether to disable policy sampling.
    /// - Returns: The selected typed policy action.
    public func act(
        observation: TensorValue<Float>,
        deterministic: Bool = true
    ) -> PolicyAction {
        act(observation: observation.elements, deterministic: deterministic)
    }
}

extension SAC {
    /// Selects an action from a typed tensor observation.
    ///
    /// - Parameters:
    ///   - observation: The typed float observation.
    ///   - deterministic: Whether to disable policy sampling.
    /// - Returns: The selected typed policy action.
    public func act(
        observation: TensorValue<Float>,
        deterministic: Bool = true
    ) -> PolicyAction {
        act(observation: observation.elements, deterministic: deterministic)
    }
}

extension TD3 {
    /// Selects an action from a typed tensor observation.
    ///
    /// - Parameters:
    ///   - observation: The typed float observation.
    ///   - deterministic: Whether to disable policy sampling.
    /// - Returns: The selected typed policy action.
    public func act(
        observation: TensorValue<Float>,
        deterministic: Bool = true
    ) -> PolicyAction {
        act(observation: observation.elements, deterministic: deterministic)
    }
}

extension PolicyAction {
    /// Converts the policy action to the typed space representation.
    ///
    /// - Returns: The corresponding type-erased space value.
    public var spaceValue: SpaceValue {
        switch self {
        case .discrete(let value):
            return .discrete(value)
        case .multiDiscrete(let values):
            return .multiDiscrete(values)
        case .multiBinary(let values):
            return .multiBinary(values)
        case .continuous(let values):
            return .continuous(values, shape: [values.count])
        }
    }
}

/// Converts a policy action into an environment-specific typed action.
public protocol PolicyActionValue: Sendable {
    /// Converts a policy action into the conforming action type.
    ///
    /// - Parameter action: The policy action to convert.
    /// - Returns: The conforming environment action.
    /// - Throws: If the policy action case does not match the conforming type.
    static func policyValue(from action: PolicyAction) throws -> Self
}

extension Int: PolicyActionValue {
    /// Extracts a discrete policy action.
    ///
    /// - Parameter action: The policy action to convert.
    /// - Returns: The discrete action value.
    /// - Throws: If action is not discrete.
    public static func policyValue(from action: PolicyAction) throws -> Int {
        guard case .discrete(let value) = action else {
            throw GymnazoError.invalidActionType(
                expected: "discrete",
                actual: String(describing: action)
            )
        }
        return value
    }
}

extension TensorValue<Float>: PolicyActionValue {
    /// Extracts a continuous policy action as a flat tensor.
    ///
    /// - Parameter action: The policy action to convert.
    /// - Returns: The continuous action as a one-dimensional tensor.
    /// - Throws: If action is not continuous or tensor construction fails.
    public static func policyValue(
        from action: PolicyAction
    ) throws -> TensorValue<Float> {
        guard case .continuous(let values) = action else {
            throw GymnazoError.invalidActionType(
                expected: "continuous",
                actual: String(describing: action)
            )
        }
        return try TensorValue(values, shape: [values.count])
    }
}

extension ValueCollectionSession
where
    Environment.Observation == TensorValue<Float>,
    Environment.Action: PolicyActionValue
{
    /// Collects transitions by asking a policy for each observation batch.
    ///
    /// - Parameters:
    ///   - steps: Number of vector frames to collect.
    ///   - seed: Optional deterministic reset seed.
    ///   - policy: Batched policy callback returning one action per slot.
    /// - Returns: The typed collection result.
    /// - Throws: If the policy returns the wrong action count or collection fails.
    public func collectPolicy(
        steps: Int,
        seed: UInt64? = nil,
        policy: @escaping @Sendable ([TensorValue<Float>]) async throws -> [PolicyAction]
    ) async throws -> ValueCollection<TensorValue<Float>, Environment.Action> {
        try await collect(steps: steps, seed: seed) { observations in
            let actions = try await policy(observations)
            guard actions.count == observations.count else {
                throw GymnazoError.vectorEnvActionCountMismatch(
                    expected: observations.count,
                    actual: actions.count
                )
            }
            return try actions.map(Environment.Action.policyValue)
        }
    }
}

extension DQN {
    /// Collects typed transitions with DQN actions.
    ///
    /// - Parameters:
    ///   - session: Exclusive typed collection session.
    ///   - steps: Number of vector frames to collect.
    ///   - seed: Optional deterministic reset seed.
    ///   - deterministic: Whether to disable policy sampling.
    /// - Returns: The collected tensor observations and integer actions.
    /// - Throws: If collection or policy inference fails.
    public func collect<Environment: ValueEnvironment>(
        from session: ValueCollectionSession<Environment>,
        steps: Int,
        seed: UInt64? = nil,
        deterministic: Bool = false
    ) async throws -> ValueCollection<TensorValue<Float>, Int>
    where
        Environment.Observation == TensorValue<Float>,
        Environment.Action == Int
    {
        try await session.collectPolicy(steps: steps, seed: seed) { observations in
            try await self.act(observations: observations, deterministic: deterministic)
        }
    }
}

extension PPO {
    /// Collects typed transitions with PPO actions.
    ///
    /// - Parameters:
    ///   - session: Exclusive typed collection session.
    ///   - steps: Number of vector frames to collect.
    ///   - seed: Optional deterministic reset seed.
    ///   - deterministic: Whether to disable policy sampling.
    /// - Returns: The collected tensor observations and typed actions.
    /// - Throws: If collection or policy inference fails.
    public func collect<Environment: ValueEnvironment>(
        from session: ValueCollectionSession<Environment>,
        steps: Int,
        seed: UInt64? = nil,
        deterministic: Bool = false
    ) async throws -> ValueCollection<TensorValue<Float>, Environment.Action>
    where
        Environment.Observation == TensorValue<Float>,
        Environment.Action: PolicyActionValue
    {
        try await session.collectPolicy(steps: steps, seed: seed) { observations in
            try await self.act(observations: observations, deterministic: deterministic)
        }
    }
}

extension SAC {
    /// Collects typed transitions with SAC actions.
    ///
    /// - Parameters:
    ///   - session: Exclusive typed collection session.
    ///   - steps: Number of vector frames to collect.
    ///   - seed: Optional deterministic reset seed.
    ///   - deterministic: Whether to disable policy sampling.
    /// - Returns: The collected tensor observations and continuous actions.
    /// - Throws: If collection or policy inference fails.
    public func collect<Environment: ValueEnvironment>(
        from session: ValueCollectionSession<Environment>,
        steps: Int,
        seed: UInt64? = nil,
        deterministic: Bool = false
    ) async throws -> ValueCollection<TensorValue<Float>, TensorValue<Float>>
    where
        Environment.Observation == TensorValue<Float>,
        Environment.Action == TensorValue<Float>
    {
        try await session.collectPolicy(steps: steps, seed: seed) { observations in
            try await self.act(observations: observations, deterministic: deterministic)
        }
    }
}

extension TD3 {
    /// Collects typed transitions with TD3 actions.
    ///
    /// - Parameters:
    ///   - session: Exclusive typed collection session.
    ///   - steps: Number of vector frames to collect.
    ///   - seed: Optional deterministic reset seed.
    ///   - deterministic: Whether to disable policy sampling.
    /// - Returns: The collected tensor observations and continuous actions.
    /// - Throws: If collection or policy inference fails.
    public func collect<Environment: ValueEnvironment>(
        from session: ValueCollectionSession<Environment>,
        steps: Int,
        seed: UInt64? = nil,
        deterministic: Bool = false
    ) async throws -> ValueCollection<TensorValue<Float>, TensorValue<Float>>
    where
        Environment.Observation == TensorValue<Float>,
        Environment.Action == TensorValue<Float>
    {
        try await session.collectPolicy(steps: steps, seed: seed) { observations in
            try await self.act(observations: observations, deterministic: deterministic)
        }
    }
}
