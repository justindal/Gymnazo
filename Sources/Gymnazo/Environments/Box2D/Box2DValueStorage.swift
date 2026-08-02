final class LunarLanderValueStorage: AnyValueEnvironmentStorage {
    private var base: LunarLander

    init(_ base: consuming LunarLander) {
        self.base = base
    }

    func reset(
        seed: UInt64?,
        options: ValueResetOptions?,
        isolation: isolated (any Actor)?
    ) async throws -> ValueReset<SpaceValue> {
        let result = try base.resetValues(seed: seed, options: options)
        return ValueReset(observation: .floatTensor(result.observation), info: result.info)
    }

    func step(
        _ action: SpaceValue,
        isolation: isolated (any Actor)?
    ) async throws -> ValueStep<SpaceValue> {
        guard case .discrete(let action) = action else {
            throw GymnazoError.invalidActionType(
                expected: "discrete action",
                actual: String(describing: action)
            )
        }
        let result = try base.stepValues(action)
        return ValueStep(
            observation: .floatTensor(result.observation),
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

final class LunarLanderContinuousValueStorage: AnyValueEnvironmentStorage {
    private var base: LunarLanderContinuous

    init(_ base: consuming LunarLanderContinuous) {
        self.base = base
    }

    func reset(
        seed: UInt64?,
        options: ValueResetOptions?,
        isolation: isolated (any Actor)?
    ) async throws -> ValueReset<SpaceValue> {
        let result = try base.resetValues(seed: seed, options: options)
        return ValueReset(observation: .floatTensor(result.observation), info: result.info)
    }

    func step(
        _ action: SpaceValue,
        isolation: isolated (any Actor)?
    ) async throws -> ValueStep<SpaceValue> {
        let values: [Float]
        switch action {
        case .floatTensor(let action):
            values = action.elements
        case .continuous(let elements, _):
            values = elements
        default:
            throw GymnazoError.invalidActionType(
                expected: "continuous action",
                actual: String(describing: action)
            )
        }
        let result = try base.stepValues(values)
        return ValueStep(
            observation: .floatTensor(result.observation),
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

final class CarRacingValueStorage: AnyValueEnvironmentStorage {
    private var base: CarRacing

    init(_ base: consuming CarRacing) {
        self.base = base
    }

    func reset(
        seed: UInt64?,
        options: ValueResetOptions?,
        isolation: isolated (any Actor)?
    ) async throws -> ValueReset<SpaceValue> {
        let randomize = try resetRandomize(options)
        let result = try base.resetValues(
            seed: seed,
            randomize: randomize,
            includeObservation: false
        )
        return ValueReset(
            observation: .floatTensor(try await pixelTensor(for: base.currentSnapshot)),
            info: result.info
        )
    }

    func step(
        _ action: SpaceValue,
        isolation: isolated (any Actor)?
    ) async throws -> ValueStep<SpaceValue> {
        let values: [Float]
        switch action {
        case .floatTensor(let action):
            values = action.elements
        case .continuous(let elements, _):
            values = elements
        default:
            throw GymnazoError.invalidActionType(
                expected: "continuous action",
                actual: String(describing: action)
            )
        }
        let result = try base.stepValues(values, includeObservation: false)
        return ValueStep(
            observation: .floatTensor(try await pixelTensor(for: base.currentSnapshot)),
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

final class CarRacingDiscreteValueStorage: AnyValueEnvironmentStorage {
    private var base: CarRacingDiscrete

    init(_ base: consuming CarRacingDiscrete) {
        self.base = base
    }

    func reset(
        seed: UInt64?,
        options: ValueResetOptions?,
        isolation: isolated (any Actor)?
    ) async throws -> ValueReset<SpaceValue> {
        let randomize = try resetRandomize(options)
        let result = try base.resetValues(
            seed: seed,
            randomize: randomize,
            includeObservation: false
        )
        return ValueReset(
            observation: .floatTensor(try await pixelTensor(for: base.currentSnapshot)),
            info: result.info
        )
    }

    func step(
        _ action: SpaceValue,
        isolation: isolated (any Actor)?
    ) async throws -> ValueStep<SpaceValue> {
        guard case .discrete(let action) = action else {
            throw GymnazoError.invalidActionType(
                expected: "discrete action",
                actual: String(describing: action)
            )
        }
        let result = try base.stepValues(action, includeObservation: false)
        return ValueStep(
            observation: .floatTensor(try await pixelTensor(for: base.currentSnapshot)),
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

private func pixelTensor(for snapshot: CarRacingSnapshot?) async throws -> TensorValue<Float> {
    let pixels: [UInt8]
    #if canImport(SwiftUI)
        if let snapshot,
            let rendered = await CarRacingRenderer.renderObservationBytes(snapshot: snapshot)
        {
            pixels = rendered
        } else {
            pixels = [UInt8](repeating: 128, count: 96 * 96 * 3)
        }
    #else
        pixels = [UInt8](repeating: 128, count: 96 * 96 * 3)
    #endif
    return try TensorValue(pixels.map(Float.init), shape: [96, 96, 3])
}

private func resetRandomize(_ options: ValueResetOptions?) throws -> Bool? {
    guard let options, !options.isEmpty else { return nil }
    guard Set(options.keys) == ["randomize"], case .bool(let randomize) = options["randomize"]
    else {
        throw GymnazoError.invalidConfiguration(
            "CarRacing reset options support only the randomize Boolean."
        )
    }
    return randomize
}
