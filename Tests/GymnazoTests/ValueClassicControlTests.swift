import MLX
import Testing

@testable import Gymnazo

@Suite("Typed classic control environments")
struct ValueClassicControlTests {
    @Test
    func cartPoleMatchesLegacySeededTrace() throws {
        var legacy = CartPole()
        var typed = try CartPoleValueEnvironment()
        let legacyReset = try legacy.reset(seed: 17)
        let typedReset = try typed.resetValue(seed: 17)

        expectClose(legacyReset.obs.asArray(Float.self), typedReset.observation.elements)

        for action in [0, 1, 1, 0, 1, 0] {
            let legacyStep = try legacy.step(MLXArray(Int32(action)))
            let typedStep = try typed.stepValue(action)
            expectClose(legacyStep.obs.asArray(Float.self), typedStep.observation.elements)
            #expect(legacyStep.reward == typedStep.reward)
            #expect(legacyStep.terminated == typedStep.terminated)
            #expect(legacyStep.truncated == typedStep.truncated)
        }
    }

    @Test
    func mountainCarMatchesLegacySeededTrace() throws {
        var legacy = MountainCar()
        var typed = try MountainCarValueEnvironment()
        let legacyReset = try legacy.reset(seed: 23)
        let typedReset = try typed.resetValue(seed: 23)

        expectClose(legacyReset.obs.asArray(Float.self), typedReset.observation.elements)

        for action in [0, 2, 2, 1, 0, 2] {
            let legacyStep = try legacy.step(MLXArray(Int32(action)))
            let typedStep = try typed.stepValue(action)
            expectClose(legacyStep.obs.asArray(Float.self), typedStep.observation.elements)
            #expect(legacyStep.reward == typedStep.reward)
            #expect(legacyStep.terminated == typedStep.terminated)
            #expect(legacyStep.truncated == typedStep.truncated)
        }
    }

    @Test
    func typedCartPoleRunsThroughParallelCollection() async throws {
        let environments = try ParallelValueVectorEnvironment<CartPoleValueEnvironment>(
            factories: [{ try CartPoleValueEnvironment() }, { try CartPoleValueEnvironment() }]
        )

        let reset = try await environments.reset(seed: 31)
        let step = try await environments.step([0, 1])

        #expect(reset.observations.count == 2)
        #expect(step.observations.count == 2)
        #expect(step.rewards == [1, 1])
        #expect(
            step.observations.allSatisfy { observation in
                observation.shape == [4]
            })
    }

    @Test
    func pendulumUsesSwiftPhysicsAndDeterministicReset() throws {
        var first = try PendulumValueEnvironment()
        var second = try PendulumValueEnvironment()

        let firstReset = try first.resetValue(seed: 23)
        let secondReset = try second.resetValue(seed: 23)
        let action = try TensorValue([Float(0.75)], shape: [1])
        let firstStep = try first.stepValue(action)
        let secondStep = try second.stepValue(action)

        #expect(firstReset.observation == secondReset.observation)
        #expect(firstStep.observation == secondStep.observation)
        #expect(firstStep.reward == secondStep.reward)
        #expect(first.valueObservationSpace.containsValue(firstStep.observation))
    }

    @Test
    func acrobotMatchesLegacyFromDeterministicState() throws {
        var legacy = Acrobot()
        var typed = try AcrobotValueEnvironment()
        let legacyReset = try legacy.reset(
            seed: 29,
            options: ["low": Float(0), "high": Float(0)]
        )
        let typedReset = try typed.resetValue(
            seed: 29,
            options: ["low": .double(0), "high": .double(0)]
        )

        expectClose(legacyReset.obs.asArray(Float.self), typedReset.observation.elements)

        for action in [0, 1, 2, 2, 1, 0] {
            let legacyStep = try legacy.step(MLXArray(Int32(action)))
            let typedStep = try typed.stepValue(action)
            expectClose(legacyStep.obs.asArray(Float.self), typedStep.observation.elements)
            #expect(abs(legacyStep.reward - typedStep.reward) < 0.000_001)
            #expect(legacyStep.terminated == typedStep.terminated)
        }
    }

    @Test
    func continuousMountainCarUsesDeterministicSwiftState() throws {
        var first = try MountainCarContinuousValueEnvironment()
        var second = try MountainCarContinuousValueEnvironment()
        let firstReset = try first.resetValue(seed: 37)
        let secondReset = try second.resetValue(seed: 37)
        let actions: [Float] = [-1, 0.5, 1, -0.25]

        #expect(firstReset.observation == secondReset.observation)
        for action in actions {
            let value = try TensorValue([action], shape: [1])
            let firstStep = try first.stepValue(value)
            let secondStep = try second.stepValue(value)
            #expect(firstStep.observation == secondStep.observation)
            #expect(firstStep.reward == secondStep.reward)
            #expect(firstStep.terminated == secondStep.terminated)
        }
    }

    private func expectClose(_ lhs: [Float], _ rhs: [Float]) {
        #expect(lhs.count == rhs.count)
        for (left, right) in zip(lhs, rhs) {
            #expect(abs(left - right) < 0.000_001)
        }
    }
}
