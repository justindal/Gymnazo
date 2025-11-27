//
//  SAC.swift
//  Soft Actor-Critic for continuous action spaces
//  Based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
//

import Collections
import Foundation
import MLXOptimizers
import MLX
import MLXNN

public class SoftQNetwork: Module {
    @ModuleInfo var layer1: Linear
    @ModuleInfo var layer2: Linear
    @ModuleInfo var layer3: Linear

    public init(numObservations: Int, numActions: Int, hiddenSize: Int = 256) {
        self._layer1.wrappedValue = Linear(numObservations + numActions, hiddenSize)
        self._layer2.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._layer3.wrappedValue = Linear(hiddenSize, 1)
        super.init()
    }

    public func callAsFunction(obs: MLXArray, action: MLXArray) -> MLXArray {
        let x = concatenated([obs, action], axis: -1)
        var h = relu(layer1(x))
        h = relu(layer2(h))
        return layer3(h)
    }
}

/// Q networks for vectorized computation using vmap.
/// This allows computing Q1 and Q2 (or more) in parallel in a single forward pass.
/// - `vmap` maps the forward function over axis 0 of weights
/// - Input `x` is broadcast (same input for all ensemble members)
/// - Output is stacked: `[numEnsemble, batch, 1]`
public class EnsembleQNetwork: Module {
    @ModuleInfo var layer1: Linear
    @ModuleInfo var layer2: Linear  
    @ModuleInfo var layer3: Linear
    
    public let numEnsemble: Int
    public let hiddenSize: Int
    
    private var vmappedForward: (([MLXArray]) -> [MLXArray])?
    
    public init(numObservations: Int, numActions: Int, numEnsemble: Int = 2, hiddenSize: Int = 256) {
        self.numEnsemble = numEnsemble
        self.hiddenSize = hiddenSize
        
        let inputSize = numObservations + numActions
        
        let bound1 = Float(sqrt(6.0 / Float(inputSize + hiddenSize)))
        let w1 = MLX.uniform(low: -bound1, high: bound1, [numEnsemble, hiddenSize, inputSize])
        let b1 = MLX.zeros([numEnsemble, hiddenSize])
        self._layer1.wrappedValue = Linear(weight: w1, bias: b1)
        
        let bound2 = Float(sqrt(6.0 / Float(hiddenSize + hiddenSize)))
        let w2 = MLX.uniform(low: -bound2, high: bound2, [numEnsemble, hiddenSize, hiddenSize])
        let b2 = MLX.zeros([numEnsemble, hiddenSize])
        self._layer2.wrappedValue = Linear(weight: w2, bias: b2)
        
        let bound3 = Float(sqrt(6.0 / Float(hiddenSize + 1)))
        let w3 = MLX.uniform(low: -bound3, high: bound3, [numEnsemble, 1, hiddenSize])
        let b3 = MLX.zeros([numEnsemble, 1])
        self._layer3.wrappedValue = Linear(weight: w3, bias: b3)
        
        super.init()
    }
    
    /// Single forward pass to be vmap'd
    private static func singleForward(arrays: [MLXArray]) -> [MLXArray] {
        let x = arrays[0]
        let w1 = arrays[1]
        let b1 = arrays[2]
        let w2 = arrays[3]
        let b2 = arrays[4]
        let w3 = arrays[5]
        let b3 = arrays[6]
        
        var h = matmul(x, w1.transposed()) + b1
        h = relu(h)
        h = matmul(h, w2.transposed()) + b2
        h = relu(h)
        let out = matmul(h, w3.transposed()) + b3
        return [out]
    }
    
    private func getVmappedForward() -> ([MLXArray]) -> [MLXArray] {
        if let existing = vmappedForward {
            return existing
        }
        
        // vmap over axis 0 of weights (ensemble dim), broadcast x (nil)
        // inAxes: [nil, 0, 0, 0, 0, 0, 0] means:
        //   - x (arrays[0]): broadcast (nil)
        //   - w1,b1,w2,b2,w3,b3: map over axis 0
        let mapped = vmap(
            EnsembleQNetwork.singleForward,
            inAxes: [nil, 0, 0, 0, 0, 0, 0],
            outAxes: [0]
        )
        vmappedForward = mapped
        return mapped
    }
    
    /// Compute Q values for all members
    /// - Parameters:
    ///   - obs: Observations [batch, obs_size]
    ///   - action: Actions [batch, action_size]
    /// - Returns: Q values [numEnsemble, batch, 1]
    public func callAsFunction(obs: MLXArray, action: MLXArray) -> MLXArray {
        let x = concatenated([obs, action], axis: -1)
        
        let w1 = layer1.weight
        let b1 = layer1.bias!
        let w2 = layer2.weight
        let b2 = layer2.bias!
        let w3 = layer3.weight
        let b3 = layer3.bias!
        
        let vf = getVmappedForward()
        let results = vf([x, w1, b1, w2, b2, w3, b3])
        return results[0]
    }
    
    /// Get minimum Q value across network members
    public func minQ(obs: MLXArray, action: MLXArray) -> MLXArray {
        let allQ = self.callAsFunction(obs: obs, action: action)
        return allQ.min(axis: 0)
    }
}

public class SACActorNetwork: Module {
    @ModuleInfo var layer1: Linear
    @ModuleInfo var layer2: Linear
    @ModuleInfo var meanLayer: Linear
    @ModuleInfo var logStdLayer: Linear

    public let actionScale: MLXArray
    public let actionBias: MLXArray
    
    private let logStdMax: Float = 2.0
    private let logStdMin: Float = -5.0
    private let logStdMinArray: MLXArray
    private let logStdRangeHalf: MLXArray
    private let logPiConstant: MLXArray
    private let epsilon: MLXArray

    public init(
        numObservations: Int,
        numActions: Int,
        hiddenSize: Int = 256,
        actionSpaceLow: Float = -1.0,
        actionSpaceHigh: Float = 1.0
    ) {
        self._layer1.wrappedValue = Linear(numObservations, hiddenSize)
        self._layer2.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._meanLayer.wrappedValue = Linear(hiddenSize, numActions)
        self._logStdLayer.wrappedValue = Linear(hiddenSize, numActions)

        let scale = (actionSpaceHigh - actionSpaceLow) / 2.0
        let bias = (actionSpaceHigh + actionSpaceLow) / 2.0
        self.actionScale = MLXArray(scale)
        self.actionBias = MLXArray(bias)
        
        self.logStdMinArray = MLXArray(logStdMin)
        self.logStdRangeHalf = MLXArray(0.5 * (logStdMax - logStdMin))
        self.logPiConstant = MLXArray(Float.log(2.0 * Float.pi))
        self.epsilon = MLXArray(Float(1e-6))
        
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> (mean: MLXArray, logStd: MLXArray) {
        var h = relu(layer1(x))
        h = relu(layer2(h))
        let mean = meanLayer(h)
        var logStd = logStdLayer(h)
        logStd = tanh(logStd)
        logStd = logStdMinArray + logStdRangeHalf * (logStd + 1.0)
        return (mean, logStd)
    }

    public func sample(obs: MLXArray, key: MLXArray) -> (action: MLXArray, logProb: MLXArray, mean: MLXArray) {
        let (mean, logStd) = self(obs)
        let std = exp(logStd)

        let noise = MLX.normal(mean.shape, key: key)
        let x_t = mean + std * noise
        let y_t = tanh(x_t)

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
