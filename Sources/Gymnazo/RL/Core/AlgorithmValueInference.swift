extension DQN {
    public func act(
        observation: TensorValue<Float>,
        deterministic: Bool = true
    ) -> PolicyAction {
        act(observation: observation.elements, deterministic: deterministic)
    }
}

extension PPO {
    public func act(
        observation: TensorValue<Float>,
        deterministic: Bool = true
    ) -> PolicyAction {
        act(observation: observation.elements, deterministic: deterministic)
    }
}

extension SAC {
    public func act(
        observation: TensorValue<Float>,
        deterministic: Bool = true
    ) -> PolicyAction {
        act(observation: observation.elements, deterministic: deterministic)
    }
}

extension TD3 {
    public func act(
        observation: TensorValue<Float>,
        deterministic: Bool = true
    ) -> PolicyAction {
        act(observation: observation.elements, deterministic: deterministic)
    }
}

extension PolicyAction {
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

public protocol PolicyActionValue: Sendable {
    static func policyValue(from action: PolicyAction) throws -> Self
}

extension Int: PolicyActionValue {
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
