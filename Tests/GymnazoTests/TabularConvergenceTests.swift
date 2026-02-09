import Testing
import MLX
@testable import Gymnazo

@Suite("Tabular Convergence", .serialized)
struct TabularConvergenceTests {
    @Test @MainActor
    func qLearningConvergesOnNonSlipperyFrozenLake() async throws {
        let env = try FrozenLake(isSlippery: false)
        let agent = TabularAgent(
            env: env,
            updateRule: .qLearning,
            config: TabularConfig(
                learningRate: 0.8,
                gamma: 0.95,
                epsilon: 1.0,
                epsilonDecay: 0.999,
                minEpsilon: 0.01
            ),
            seed: 42
        )

        nonisolated(unsafe) var rewards: [Double] = []
        let callbacks = LearnCallbacks(
            onEpisodeEnd: { @Sendable reward, _ in
                rewards.append(reward)
            }
        )

        try await agent.learn(totalTimesteps: 50_000, callbacks: callbacks)

        let last100 = Array(rewards.suffix(100))
        let avgReward = last100.reduce(0, +) / Double(last100.count)

        #expect(rewards.count > 50)
        #expect(avgReward > 0.9)
    }

    @Test @MainActor
    func sarsaConvergesOnNonSlipperyFrozenLake() async throws {
        let env = try FrozenLake(isSlippery: false)
        let agent = TabularAgent(
            env: env,
            updateRule: .sarsa,
            config: TabularConfig(
                learningRate: 0.8,
                gamma: 0.95,
                epsilon: 1.0,
                epsilonDecay: 0.999,
                minEpsilon: 0.01
            ),
            seed: 42
        )

        nonisolated(unsafe) var rewards: [Double] = []
        let callbacks = LearnCallbacks(
            onEpisodeEnd: { @Sendable reward, _ in
                rewards.append(reward)
            }
        )

        try await agent.learn(totalTimesteps: 50_000, callbacks: callbacks)

        let last100 = Array(rewards.suffix(100))
        let avgReward = last100.reduce(0, +) / Double(last100.count)

        #expect(rewards.count > 50)
        #expect(avgReward > 0.9)
    }

    @Test @MainActor
    func qLearningLearnsOnSlipperyFrozenLake() async throws {
        let env = try FrozenLake(isSlippery: true)
        let agent = TabularAgent(
            env: env,
            updateRule: .qLearning,
            seed: 42
        )

        nonisolated(unsafe) var rewards: [Double] = []
        let callbacks = LearnCallbacks(
            onEpisodeEnd: { @Sendable reward, _ in
                rewards.append(reward)
            }
        )

        try await agent.learn(totalTimesteps: 100_000, callbacks: callbacks)

        let last500 = Array(rewards.suffix(500))
        let avgReward = last500.reduce(0, +) / Double(last500.count)

        #expect(rewards.count > 100)
        #expect(avgReward > 0.5)
    }
}
