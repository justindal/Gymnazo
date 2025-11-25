//
//  SAC.swift
//  uses logic from
//  https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
//

import MLX
import MLXNN

public class SoftQNetwork: Module {
    @ModuleInfo var layer1: Linear
    @ModuleInfo var layer2: Linear
    @ModuleInfo var layer3: Linear

    init(numObservations: Int, numActions: Int, hiddenSize: Int = 256) {
        self.layer1 = Linear(numObservations + numActions, hiddenSize)
        self.layer2 = Linear(hiddenSize, hiddenSize)
        self.layer3 = Linear(hiddenSize, 1)
    }

    public func callAsFunction(obs: MLXArray, action: MLXArray) -> MLXArray {
        let x = concatenated([obs, action], axis: -1)
        var h = relu(layer1(x))
        h = relu(layer2(h))
        return layer3(h)
    }
}

public class Actor: Module {
    @ModuleInfo var layer1: Linear
    @ModuleInfo var layer2: Linear
    @ModuleInfo var meanLayer: Linear
    @ModuleInfo var logStdLayer: Linear

    let actionScale: MLXArray
    let actionBias: MLXArray

    public init(
        numObservations: Int,
        numActions: Int,
        hiddenSize: Int = 256,
        actionSpaceLow: Float = -1.0,
        actionSpaceHigh: Float = 1.0
    ) {
        self.layer1 = Linear(numObservations, hiddenSize)
        self.layer2 = Linear(hiddenSize, hiddenSize)
        self.meanLayer = Linear(hiddenSize, numActions)
        self.logStdLayer = Linear(hiddenSize, numActions)

        let scale = (actionSpaceHigh - actionSpaceLow) / 2.0
        let bias = (actionSpaceHigh + actionSpaceLow) / 2.0
        self.actionScale = MLXArray(scale)
        self.actionBias = MLXArray(bias)
    }

    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var x = relu(layer1(x))
        x = relu(layer2(x))

        let mean = meanLayer(x)

        let LogStdMax = MLXArray(2)
        let LogStdMin = MLXArray(-5)

        var logStd = logStdLayer(x)
        logStd = tanh(logStd)
        logStd = LogStdMin + 0.5 * (LogStdMax - LogStdMin) * (logStd + 1.0)

        return (mean, logStd)
    }

    public func sample(obs: MLXArray, key: MLXArray) -> (
        action: MLXArray, logProb: MLXArray, mean: MLXArray
    ) {
        let (mean, logStd) = self(obs)
        let std = exp(logStd)

        // reparameterization trick
        let noise = std * MLX.normal(mean.shape, key: key)
        let x_t = mean + std * noise
        let y_t = tan(x_t)

        let action = y_t * actionScale + actionBias

        // enforce action bounds
        let logProbNorm =
            -0.5
            * (pow((x_t - mean) / std, 2.0)
                + 2.0 * logStd
                + log(MLXArray(2.0 * Double.pi)))

        let logProbCorrection = log(1 - pow(y_t, 2) + 1e-6)
        
        let logProb = (logProbNorm - logProbCorrection).sum(
            axis: 1,
            keepDims: true
        )

        return (action, logProb, mean)
    }
}
