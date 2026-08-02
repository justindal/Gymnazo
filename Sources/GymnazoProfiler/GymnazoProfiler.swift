import Foundation
import Gymnazo
import MLX

struct ProfileMeasurement: Codable {
    let name: String
    let operations: Int
    let seconds: Double
    let operationsPerSecond: Double
    let activeMemoryBytes: Int
    let peakMemoryBytes: Int
    let cacheMemoryBytes: Int
}

struct ProfileReport: Codable {
    let architecture: String
    let environment: String
    let vectorEnvironments: Int
    let measurements: [ProfileMeasurement]
}

enum ProfileError: Error {
    case unsupportedPlatform
}

@main
@MainActor
struct GymnazoProfiler {
    static func main() async throws {
        #if arch(arm64) && os(macOS)
            Memory.clearCache()
            let vectorEnvironments = 8
            let replay = try await profileReplay(vectorEnvironments: vectorEnvironments)
            let training = try await profileTraining()
            let report = ProfileReport(
                architecture: "arm64",
                environment: "CartPole",
                vectorEnvironments: vectorEnvironments,
                measurements: replay + training
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(report)
            guard let output = String(data: data, encoding: .utf8) else {
                throw CocoaError(.fileWriteInapplicableStringEncoding)
            }
            print(output)
        #else
            throw ProfileError.unsupportedPlatform
        #endif
    }

    private static func profileReplay(
        vectorEnvironments: Int
    ) async throws -> [ProfileMeasurement] {
        let model = try DQN(
            observationSpace: Box(low: -100, high: 100, shape: [4]),
            actionSpace: Discrete(n: 2),
            policyConfig: DQNPolicyConfig(netArch: [64, 64], featuresExtractor: .flatten),
            config: DQNConfig(explorationInitialEps: 0, explorationFinalEps: 0),
            seed: 7
        )
        let session = try ValueCollectionSession(
            environments: try (0..<vectorEnvironments).map { _ in
                try CartPoleValueEnvironment()
            }
        )

        let warmCollection = try await measured(
            name: "collectionWarmUp",
            operations: vectorEnvironments
        ) {
            try await model.collect(
                from: session,
                steps: 1,
                seed: 11,
                deterministic: true
            )
        }
        let steadySteps = 256
        let steadyCollection = try await measured(
            name: "collectionSteadyState",
            operations: steadySteps * vectorEnvironments
        ) {
            try await model.collect(
                from: session,
                steps: steadySteps,
                seed: 13,
                deterministic: true
            )
        }

        var replay = try ReplayBuffer(
            observationSpace: Box(low: -100, high: 100, shape: [4]),
            actionSpace: Discrete(n: 2),
            config: .init(
                bufferSize: (steadySteps + 1) * vectorEnvironments,
                seed: 17
            ),
            numEnvs: vectorEnvironments
        )
        let replayWarmUp = try await measured(
            name: "replayInsertWarmUp",
            operations: vectorEnvironments
        ) {
            try replay.add(warmCollection.value)
        }
        let replaySteady = try await measured(
            name: "replayInsertSteadyState",
            operations: steadySteps * vectorEnvironments
        ) {
            try replay.add(steadyCollection.value)
        }
        try await session.close()
        return [
            warmCollection.measurement,
            steadyCollection.measurement,
            replayWarmUp.measurement,
            replaySteady.measurement,
        ]
    }

    private static func profileTraining() async throws -> [ProfileMeasurement] {
        let env = CartPole()
        let model = try DQN(
            env: env,
            policyConfig: DQNPolicyConfig(netArch: [64, 64], featuresExtractor: .flatten),
            config: DQNConfig(
                bufferSize: 2_048,
                learningStarts: 32,
                batchSize: 32,
                trainFrequency: TrainFrequency(frequency: 1, unit: .step),
                gradientSteps: .fixed(1),
                targetUpdateInterval: 64,
                explorationInitialEps: 0,
                explorationFinalEps: 0
            ),
            seed: 19
        )
        let warmSteps = 64
        let warmUp = try await measured(
            name: "trainingWarmUp",
            operations: warmSteps
        ) {
            try await model.learn(totalTimesteps: warmSteps, callbacks: nil)
        }
        let steadySteps = 256
        let steady = try await measured(
            name: "trainingSteadyState",
            operations: steadySteps
        ) {
            try await model.learn(
                totalTimesteps: warmSteps + steadySteps,
                callbacks: nil,
                resetProgress: false
            )
        }
        return [warmUp.measurement, steady.measurement]
    }

    private static func measured<Value>(
        name: String,
        operations: Int,
        operation: () async throws -> Value
    ) async rethrows -> (value: Value, measurement: ProfileMeasurement) {
        Memory.peakMemory = 0
        let clock = ContinuousClock()
        let start = clock.now
        let value = try await operation()
        let duration = start.duration(to: clock.now)
        let seconds = duration.seconds
        return (
            value,
            ProfileMeasurement(
                name: name,
                operations: operations,
                seconds: seconds,
                operationsPerSecond: seconds > 0 ? Double(operations) / seconds : 0,
                activeMemoryBytes: Memory.activeMemory,
                peakMemoryBytes: Memory.peakMemory,
                cacheMemoryBytes: Memory.cacheMemory
            )
        )
    }
}

extension Duration {
    fileprivate var seconds: Double {
        let components = self.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
