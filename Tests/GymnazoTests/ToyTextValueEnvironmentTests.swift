import MLX
import Testing

@testable import Gymnazo

@Suite("Toy Text typed environment parity")
struct ToyTextValueEnvironmentTests {
    @Test
    func frozenLakeMatchesCompatibilityTrace() async throws {
        var compatibility = try FrozenLake(isSlippery: false)
        let typed = try await Gymnazo.makeValue(
            "FrozenLake",
            options: ["is_slippery": false]
        )

        try await expectDiscreteReset(&compatibility, typed, seed: 11)
        for action in [2, 2, 1, 1, 2, 2] {
            let result = try compatibility.step(MLXArray(Int32(action)))
            let value = try await typed.step(.discrete(action))
            expectDiscreteStep(result, value)
            if result.terminated { break }
        }
    }

    @Test
    func blackjackMatchesSeededCompatibilityTrace() async throws {
        var compatibility = Blackjack(natural: true)
        let typed = try await Gymnazo.makeValue(
            "Blackjack",
            options: ["natural": true]
        )

        let reset = try compatibility.reset(seed: 19)
        let valueReset = try await typed.reset(seed: 19)
        expectTuple(reset.obs, valueReset.observation)

        let result = try compatibility.step(MLXArray(Int32(0)))
        let value = try await typed.step(.discrete(0))
        expectTuple(result.obs, value.observation)
        #expect(result.reward == value.reward)
        #expect(result.terminated == value.terminated)
    }

    @Test
    func taxiMatchesCompatibilityTrace() async throws {
        var compatibility = Taxi(isRainy: true, ficklePassenger: true)
        let typed = try await Gymnazo.makeValue(
            "Taxi",
            options: ["is_rainy": true, "fickle_passenger": true]
        )

        try await expectDiscreteReset(&compatibility, typed, seed: 31)
        for action in [0, 2, 2, 1, 3, 4, 0, 5] {
            let result = try compatibility.step(MLXArray(Int32(action)))
            let value = try await typed.step(.discrete(action))
            expectDiscreteStep(result, value)
            if result.terminated { break }
        }
    }

    @Test
    func cliffWalkingMatchesCompatibilityTrace() async throws {
        var compatibility = CliffWalking(isSlippery: true)
        let typed = try await Gymnazo.makeValue(
            "CliffWalking",
            options: ["is_slippery": true]
        )

        try await expectDiscreteReset(&compatibility, typed, seed: 47)
        for action in [0, 1, 1, 2, 0, 3, 1] {
            let result = try compatibility.step(MLXArray(Int32(action)))
            let value = try await typed.step(.discrete(action))
            expectDiscreteStep(result, value)
            if result.terminated { break }
        }
    }

    private func expectDiscreteReset<Base: Env>(
        _ compatibility: inout Base,
        _ typed: AnyValueEnvironment,
        seed: UInt64
    ) async throws {
        let reset = try compatibility.reset(seed: seed)
        let value = try await typed.reset(seed: seed)
        #expect(discrete(reset.obs) == discrete(value.observation))
    }

    private func expectDiscreteStep(_ result: Step, _ value: ValueStep<SpaceValue>) {
        #expect(discrete(result.obs) == discrete(value.observation))
        #expect(result.reward == value.reward)
        #expect(result.terminated == value.terminated)
        #expect(result.truncated == value.truncated)
    }

    private func discrete(_ value: MLXArray) -> Int {
        Int(value.singletonValue(Int32.self))
    }

    private func discrete(_ value: SpaceValue) -> Int? {
        guard case .discrete(let value) = value else { return nil }
        return value
    }

    private func expectTuple(_ compatibility: MLXArray, _ typed: SpaceValue) {
        guard case .tuple(let values) = typed else {
            Issue.record("Expected a typed tuple.")
            return
        }
        let expected = compatibility.asType(.int32).asArray(Int32.self).map(Int.init)
        let actual = values.compactMap(discrete)
        #expect(actual == expected)
    }
}
