import Foundation
import MLX
import Synchronization
import Testing

@testable import Gymnazo

@Suite(.serialized)
struct TrainingPhaseTests {
    @Test
    func firstUpdateReportsCompilationAndLaterUpdatesReportTraining() async throws {
        let env = try await Gymnazo.make("CartPole")
        let ppo = try PPO(
            env: env,
            policyConfig: PPOPolicyConfig(netArch: .shared([8])),
            config: PPOConfig(nSteps: 4, batchSize: 2, nEpochs: 1), seed: 7
        )
        let phases = Mutex<[TrainingPhase]>([])
        let entropyCoefficients = Mutex<[Double]>([])
        let callbacks = LearnCallbacks(
            onTrain: { metrics in
                if let value = metrics["train/ent_coef"] { entropyCoefficients.withLock { $0.append(value) } }
            },
            onPhase: { phase in phases.withLock { $0.append(phase) } }
        )
        try await ppo.learn(totalTimesteps: 8, callbacks: callbacks)
        #expect(phases.withLock { $0 } == [.compiling, .collecting, .updating, .collecting])
        try await ppo.learn(totalTimesteps: 12, callbacks: callbacks, resetProgress: false)
        #expect(await ppo.numTimesteps == 12)
        #expect(phases.withLock { $0.filter { $0 == .compiling }.count } == 1)
        #expect(entropyCoefficients.withLock { $0 } == [0, 0, 0])
    }
}
