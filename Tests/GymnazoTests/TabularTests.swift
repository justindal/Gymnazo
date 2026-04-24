import MLX
import Testing

@testable import Gymnazo

final class NonTerminalTabularEnv: Env {
    let actionSpace: any Space = Discrete(n: 2)
    let observationSpace: any Space = Discrete(n: 2)
    var spec: EnvSpec? = nil
    var renderMode: RenderMode? = nil

    func step(_ action: MLXArray) throws -> Step {
        Step(
            obs: MLXArray(Int32(1)),
            reward: 1.0,
            terminated: false,
            truncated: false,
            info: [:]
        )
    }

    func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        Reset(obs: MLXArray(Int32(0)), info: [:])
    }
}

final class TerminalTabularEnv: Env {
    let actionSpace: any Space = Discrete(n: 2)
    let observationSpace: any Space = Discrete(n: 2)
    var spec: EnvSpec? = nil
    var renderMode: RenderMode? = nil

    func step(_ action: MLXArray) throws -> Step {
        Step(
            obs: MLXArray(Int32(1)),
            reward: 1.0,
            terminated: true,
            truncated: false,
            info: [:]
        )
    }

    func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        Reset(obs: MLXArray(Int32(0)), info: [:])
    }
}

@Suite("Tabular", .serialized)
struct TabularTests {
    @Test
    @MainActor
    func qLearningOneStepUpdateMatchesClosedForm() async throws {
        let env = NonTerminalTabularEnv()
        let config = TabularConfig(
            learningRate: 0.5,
            gamma: 0.5,
            epsilon: 0.0,
            epsilonDecay: 1.0,
            minEpsilon: 0.0
        )
        var initialQ = MLX.zeros([2, 2])
        initialQ[1, 0] = MLXArray(2.0 as Float)
        initialQ[1, 1] = MLXArray(5.0 as Float)
        eval(initialQ)
        let agent = TabularAgent(
            updateRule: .qLearning,
            config: config,
            numStates: 2,
            numActions: 2,
            seed: 12,
            qTable: initialQ,
            timesteps: 0,
            explorationRate: 0.0
        )
        await agent.setEnv(EnvBox(env))

        try await agent.learn(totalTimesteps: 1, callbacks: nil as LearnCallbacks?)
        let values = await agent.tableValues()

        let expected = Float(1.75)
        #expect(abs(values[0] - expected) < 1e-5)
        #expect(abs(values[2] - 2.0) < 1e-6)
        #expect(abs(values[3] - 5.0) < 1e-6)
    }

    @Test
    @MainActor
    func sarsaOneStepUpdateMatchesClosedForm() async throws {
        let env = NonTerminalTabularEnv()
        let config = TabularConfig(
            learningRate: 0.5,
            gamma: 0.5,
            epsilon: 0.0,
            epsilonDecay: 1.0,
            minEpsilon: 0.0
        )
        var initialQ = MLX.zeros([2, 2])
        initialQ[1, 0] = MLXArray(4.0 as Float)
        initialQ[1, 1] = MLXArray(1.0 as Float)
        eval(initialQ)
        let agent = TabularAgent(
            updateRule: .sarsa,
            config: config,
            numStates: 2,
            numActions: 2,
            seed: 21,
            qTable: initialQ,
            timesteps: 0,
            explorationRate: 0.0
        )
        await agent.setEnv(EnvBox(env))

        try await agent.learn(totalTimesteps: 1, callbacks: nil as LearnCallbacks?)
        let values = await agent.tableValues()

        let expected = Float(1.5)
        #expect(abs(values[0] - expected) < 1e-5)
        #expect(abs(values[2] - 4.0) < 1e-6)
        #expect(abs(values[3] - 1.0) < 1e-6)
    }

    @Test
    @MainActor
    func epsilonDecaysOnEpisodeEndWithFloor() async throws {
        let env = TerminalTabularEnv()
        let config = TabularConfig(
            learningRate: 0.1,
            gamma: 0.99,
            epsilon: 1.0,
            epsilonDecay: 0.5,
            minEpsilon: 0.2
        )
        let agent = TabularAgent(
            env: env,
            updateRule: .qLearning,
            config: config,
            seed: 9
        )

        try await agent.learn(totalTimesteps: 3, callbacks: nil as LearnCallbacks?)
        #expect(abs(await agent.epsilon - 0.2) < 1e-6)
    }
}
