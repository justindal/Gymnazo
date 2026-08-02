import Testing

@testable import Gymnazo

private struct EndlessValueEnvironment: ValueEnvironment {
    let valueObservationSpace: DiscreteValueSpace
    let valueActionSpace: DiscreteValueSpace

    init() throws {
        valueObservationSpace = try DiscreteValueSpace(n: 1)
        valueActionSpace = try DiscreteValueSpace(n: 1)
    }

    mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Int> {
        ValueReset(observation: 0)
    }

    mutating func stepValue(_ action: Int) throws -> ValueStep<Int> {
        ValueStep(
            observation: 0,
            reward: 1,
            terminated: false,
            truncated: false
        )
    }

    mutating func close() {}
}

@Suite("Typed environment registry")
struct ValueRegistryTests {
    @Test
    func unversionedIDsAreRegistered() async {
        let ids = await Gymnazo.valueIDs()

        #expect(ids.contains("CartPole"))
        #expect(ids.contains("MountainCar"))
        #expect(ids.contains("Pendulum"))
        #expect(ids.contains("Acrobot"))
        #expect(ids.contains("MountainCarContinuous"))
        #expect(ids.contains("FrozenLake"))
        #expect(ids.contains("Blackjack"))
        #expect(ids.contains("Taxi"))
        #expect(ids.contains("CliffWalking"))
        #expect(ids.contains("LunarLander"))
        #expect(ids.contains("LunarLanderContinuous"))
        #expect(ids.contains("CarRacing"))
        #expect(ids.contains("CarRacingDiscrete"))
    }

    @Test
    func dynamicEnvironmentPreservesValueSemantics() async throws {
        let environment = try await Gymnazo.makeValue("CartPole")
        let reset = try await environment.reset(seed: 42)
        let step = try await environment.step(.discrete(1))

        guard case .floatTensor(let observation) = reset.observation else {
            Issue.record("Expected a typed float observation.")
            return
        }

        #expect(observation.shape == [4])
        #expect(environment.observationSpace.containsValue(step.observation))

        await #expect(throws: GymnazoError.self) {
            try await environment.step(.discrete(4))
        }
    }

    @Test
    func continuousEnvironmentUsesTypedActions() async throws {
        let environment = try await Gymnazo.makeValue("Pendulum", options: ["g": 10.0])
        _ = try await environment.reset(
            seed: 42,
            options: ["x_init": .double(0), "y_init": .double(0)]
        )
        let step = try await environment.step(
            .floatTensor(try TensorValue([2], shape: [1]))
        )

        #expect(abs(step.reward + 0.004) < 0.000_001)
        #expect(environment.observationSpace.containsValue(step.observation))
    }

    @Test
    func bundledFamiliesUseActorOwnedTypedValues() async throws {
        for id in [
            "FrozenLake",
            "Blackjack",
            "Taxi",
            "CliffWalking",
            "LunarLander",
            "LunarLanderContinuous",
            "CarRacing",
            "CarRacingDiscrete",
        ] {
            let environment = try await Gymnazo.makeValue(id)
            #expect(environment.id == id)
        }

        let frozenLake = try await Gymnazo.makeValue(
            "FrozenLake",
            options: ["is_slippery": false]
        )
        #expect(frozenLake.id == "FrozenLake")
        let lakeReset = try await frozenLake.reset(seed: 5)
        #expect(frozenLake.observationSpace.containsValue(lakeReset.observation))
        let lakeStep = try await frozenLake.step(.discrete(2))
        #expect(frozenLake.observationSpace.containsValue(lakeStep.observation))

        let blackjack = try await Gymnazo.makeValue("Blackjack")
        let blackjackReset = try await blackjack.reset(seed: 5)
        guard case .tuple(let values) = blackjackReset.observation else {
            Issue.record("Expected a typed tuple observation.")
            return
        }
        #expect(values.count == 3)

        let lander = try await Gymnazo.makeValue("LunarLander")
        let landerReset = try await lander.reset(seed: 5)
        #expect(lander.observationSpace.containsValue(landerReset.observation))
        let landerStep = try await lander.step(.discrete(0))
        #expect(lander.observationSpace.containsValue(landerStep.observation))
    }

    @Test
    func canonicalAndAliasFactoriesApplyEpisodeLimits() async throws {
        let registry = ValueGymnazoRegistry()
        for id in ["Endless-v0", "Endless"] {
            await registry.register(
                ValueEnvSpec(
                    id: id,
                    entryPoint: { _ in
                        AnyValueEnvironment(id: id, try EndlessValueEnvironment())
                    },
                    maxEpisodeSteps: 2
                )
            )
        }

        for id in ["Endless-v0", "Endless"] {
            let environment = try await registry.make(id)
            _ = try await environment.reset(seed: 7)
            let first = try await environment.step(.discrete(0))
            let second = try await environment.step(.discrete(0))

            #expect(!first.truncated)
            #expect(second.truncated)
            #expect(second.info[EnvInfoKey.timeLimitTruncated] == .bool(true))
        }
    }
}
