import MLX

enum DQNMath {
    static func explorationRate(
        initial: Double,
        final: Double,
        timestep: Int,
        totalTimesteps: Int,
        fraction: Double
    ) -> Double {
        guard totalTimesteps > 0, fraction > 0 else { return final }
        let progress = min(1, Double(timestep) / (Double(totalTimesteps) * fraction))
        return initial + progress * (final - initial)
    }

    static func target(
        rewards: MLXArray,
        dones: MLXArray,
        nextQ: MLXArray,
        gamma: Float
    ) -> MLXArray {
        rewards.expandedDimensions(axis: -1)
            + (1 - dones.expandedDimensions(axis: -1)) * gamma * nextQ
    }

    static func selectedQ(_ values: MLXArray, actions: MLXArray) -> MLXArray {
        let batch = actions.shape[0]
        let indices = MLXArray(Array(0..<Int32(batch)))
        let actionIndices = actions.reshaped([-1]).asType(.int32)
        return values[indices, actionIndices].expandedDimensions(axis: -1)
    }

    static func loss(selectedQ: MLXArray, targetQ: MLXArray) -> MLXArray {
        MLX.mean((selectedQ - targetQ) ** 2)
    }
}

enum PPOMath {
    static func losses(
        values: MLXArray,
        logProb: MLXArray,
        entropy: MLXArray?,
        oldValues: MLXArray,
        oldLogProbs: MLXArray,
        advantages: MLXArray,
        returns: MLXArray,
        clipRange: Float,
        clipRangeVf: Float?,
        entCoef: Float,
        vfCoef: Float
    ) -> [MLXArray] {
        let ratio = MLX.exp(logProb - oldLogProbs)
        let unclipped = advantages * ratio
        let clipped =
            advantages
            * MLX.clip(
                ratio,
                min: 1 - clipRange,
                max: 1 + clipRange
            )
        let policyLoss = -MLX.mean(MLX.minimum(unclipped, clipped))

        let valuePred: MLXArray
        if let clipRangeVf {
            valuePred =
                oldValues
                + MLX.clip(
                    values - oldValues,
                    min: -clipRangeVf,
                    max: clipRangeVf
                )
        } else {
            valuePred = values
        }
        let valueLoss = MLX.mean((returns - valuePred) ** 2)
        let entropyLoss = entropy.map { -MLX.mean($0) } ?? MLX.mean(logProb)
        let totalLoss = policyLoss + entCoef * entropyLoss + vfCoef * valueLoss
        let logRatio = logProb - oldLogProbs
        let approxKL = MLX.mean((MLX.exp(logRatio) - 1) - logRatio)
        let clipFraction = MLX.mean(
            (MLX.abs(ratio - 1) .> clipRange).asType(.float32)
        )
        return [totalLoss, policyLoss, valueLoss, entropyLoss, approxKL, clipFraction]
    }

    static func shouldStop(approxKL: Float, targetKL: Double?) -> Bool {
        guard let targetKL else { return false }
        return approxKL > Float(1.5 * targetKL)
    }

    static func reward(
        _ reward: Float,
        terminated: Bool,
        truncated: Bool,
        gamma: Double,
        terminalValue: Float
    ) -> Float {
        shouldBootstrap(terminated: terminated, truncated: truncated)
            ? reward + Float(gamma) * terminalValue : reward
    }

    static func shouldBootstrap(terminated: Bool, truncated: Bool) -> Bool {
        truncated && !terminated
    }
}

enum SACMath {
    static func target(
        rewards: MLXArray,
        dones: MLXArray,
        minQ: MLXArray,
        nextLogProb: MLXArray,
        entCoef: MLXArray,
        gamma: Float
    ) -> MLXArray {
        let nextQ = minQ - entCoef * nextLogProb.expandedDimensions(axis: -1)
        return rewards.expandedDimensions(axis: -1)
            + (1 - dones.expandedDimensions(axis: -1)) * gamma * nextQ
    }

    static func entropyLoss(
        logEntCoef: MLXArray,
        logProb: MLXArray,
        targetEntropy: Float
    ) -> MLXArray {
        MLX.mean(-logEntCoef * (logProb + targetEntropy))
    }
}

enum TD3Math {
    static func targetActions(
        _ actions: MLXArray,
        noise: MLXArray,
        noiseClip: Float
    ) -> MLXArray {
        let clippedNoise = MLX.clip(noise, min: -noiseClip, max: noiseClip)
        return MLX.clip(actions + clippedNoise, min: -1, max: 1)
    }

    static func target(
        rewards: MLXArray,
        dones: MLXArray,
        minQ: MLXArray,
        gamma: Float
    ) -> MLXArray {
        rewards.expandedDimensions(axis: -1)
            + (1 - dones.expandedDimensions(axis: -1)) * gamma * minQ
    }

    static func shouldUpdateActor(step: Int, delay: Int) -> Bool {
        delay > 0 && step.isMultiple(of: delay)
    }
}
