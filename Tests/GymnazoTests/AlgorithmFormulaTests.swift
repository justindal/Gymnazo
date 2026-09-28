import Foundation
import MLX
import MLXNN
import Testing

@testable import Gymnazo

@Suite("Algorithm formula contracts")
struct AlgorithmFormulaTests {
    @Test
    func dqnBellmanGatherLossAndSchedule() {
        let qValues = MLXArray([Float(1), 5, 7, 2], [2, 2])
        let actions = MLXArray([Int32(1), 0], [2, 1])
        let selected = DQNMath.selectedQ(qValues, actions: actions)
        let target = DQNMath.target(
            rewards: MLXArray([Float(1), 2]),
            dones: MLXArray([Float(0), 1]),
            nextQ: MLXArray([Float(3), 9], [2, 1]),
            gamma: 0.9
        )
        let loss = DQNMath.loss(
            selectedQ: selected,
            targetQ: MLXArray([Float(4), 6], [2, 1])
        )
        eval(selected, target, loss)

        #expect(selected.asArray(Float.self) == [5, 7])
        let targets = target.asArray(Float.self)
        #expect(abs(targets[0] - 3.7) < 0.000_001)
        #expect(targets[1] == 2)
        #expect(abs(loss.item(Float.self) - 1) < 0.000_001)
        let midpoint = DQNMath.explorationRate(
            initial: 1,
            final: 0.1,
            timestep: 25,
            totalTimesteps: 100,
            fraction: 0.5
        )
        let endpoint = DQNMath.explorationRate(
            initial: 1,
            final: 0.1,
            timestep: 100,
            totalTimesteps: 100,
            fraction: 0.5
        )
        #expect(abs(midpoint - 0.55) < 0.000_001)
        #expect(abs(endpoint - 0.1) < 0.000_001)
    }

    @Test
    func ppoClippedObjectiveValueEntropyAndKL() {
        let newLogProb = MLXArray([Float(log(1.5)), Float(log(0.5))])
        let losses = PPOMath.losses(
            values: MLXArray([Float(3), 0]),
            logProb: newLogProb,
            entropy: MLXArray([Float(0.7), 0.5]),
            oldValues: MLXArray([Float(1), 1]),
            oldLogProbs: MLXArray([Float(0), 0]),
            advantages: MLXArray([Float(1), -1]),
            returns: MLXArray([Float(2), 1]),
            clipRange: 0.2,
            clipRangeVf: 0.5,
            entCoef: 0.01,
            vfCoef: 0.5
        )
        eval(losses)
        let values = losses.map { $0.item(Float.self) }
        let expectedKL = Float(((1.5 - 1 - log(1.5)) + (0.5 - 1 - log(0.5))) / 2)

        #expect(abs(values[0] + 0.081) < 0.000_001)
        #expect(abs(values[1] + 0.2) < 0.000_001)
        #expect(abs(values[2] - 0.25) < 0.000_001)
        #expect(abs(values[3] + 0.6) < 0.000_001)
        #expect(abs(values[4] - expectedKL) < 0.000_001)
        #expect(abs(values[5] - 1) < 0.000_001)
        #expect(PPOMath.shouldStop(approxKL: 0.031, targetKL: 0.02))
        #expect(!PPOMath.shouldStop(approxKL: 0.03, targetKL: 0.02))
        #expect(
            PPOMath.reward(1, terminated: false, truncated: false, gamma: 0.9, terminalValue: 4)
                == 1)
        #expect(
            abs(
                PPOMath.reward(1, terminated: false, truncated: true, gamma: 0.9, terminalValue: 4)
                    - 4.6) < 0.000_001)
        #expect(
            PPOMath.reward(1, terminated: true, truncated: false, gamma: 0.9, terminalValue: 4) == 1
        )
        #expect(
            PPOMath.reward(1, terminated: true, truncated: true, gamma: 0.9, terminalValue: 4) == 1)
    }

    @Test
    func sacTargetAndEntropyTemperature() {
        let target = SACMath.target(
            rewards: MLXArray([Float(1), 2]),
            dones: MLXArray([Float(0), 1]),
            minQ: MLXArray([Float(5), 7], [2, 1]),
            nextLogProb: MLXArray([Float(-0.5), -0.25]),
            entCoef: MLXArray(Float(0.2)),
            gamma: 0.9
        )
        let entropyLoss = SACMath.entropyLoss(
            logEntCoef: MLXArray(Float(log(0.2))),
            logProb: MLXArray([Float(-0.5), -1.5]),
            targetEntropy: -2
        )
        eval(target, entropyLoss)
        let values = target.asArray(Float.self)

        #expect(abs(values[0] - 5.59) < 0.000_001)
        #expect(values[1] == 2)
        #expect(abs(entropyLoss.item(Float.self) - Float(3 * log(0.2))) < 0.000_001)
    }

    @Test
    func td3NoiseDelayTargetAndPolyak() {
        let actions = TD3Math.targetActions(
            MLXArray([Float(0.9), -0.9], [1, 2]),
            noise: MLXArray([Float(0.5), -0.5], [1, 2]),
            noiseClip: 0.2
        )
        let target = TD3Math.target(
            rewards: MLXArray([Float(1), 2]),
            dones: MLXArray([Float(0), 1]),
            minQ: MLXArray([Float(3), 8], [2, 1]),
            gamma: 0.5
        )
        let updated = polyakUpdate(
            target: ModuleParameters.unflattened(["weight": MLXArray([Float(0), 2])]),
            source: ModuleParameters.unflattened(["weight": MLXArray([Float(2), 4])]),
            tau: 0.25
        )
        guard let weights = updated.flattened().first(where: { $0.0 == "weight" })?.1 else {
            Issue.record("Missing Polyak-updated weight.")
            return
        }
        eval(actions, target, weights)

        #expect(actions.asArray(Float.self) == [1, -1])
        #expect(target.asArray(Float.self) == [2.5, 2])
        #expect(TD3Math.shouldUpdateActor(step: 2, delay: 2))
        #expect(!TD3Math.shouldUpdateActor(step: 3, delay: 2))
        #expect(weights.asArray(Float.self) == [0.5, 2.5])
    }

    @Test
    func zeroTD3NoiseClipRemovesTargetNoise() {
        let action = TD3Math.targetActions(
            MLXArray([Float(0.2)]), noise: MLXArray([Float(0.5)]), noiseClip: 0
        )
        eval(action)
        #expect(action.asArray(Float.self) == [0.2])
    }

    @Test
    func explainedVarianceIgnoresConstantPredictionBias() {
        #expect(PPO.explainedVariance(values: [0, 1, 2], returns: [1, 2, 3]) == 1)
        #expect(PPO.explainedVariance(values: [1, 2, 3], returns: [1, 2, 3]) == 1)
        #expect(PPO.explainedVariance(values: [3, 2, 1], returns: [1, 2, 3]) == -3)
        #expect(PPO.explainedVariance(values: [1, 1, 1], returns: [1, 1, 1]) == 0)
    }
}
