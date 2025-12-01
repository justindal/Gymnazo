//
//  SAC.swift
//  Soft Actor-Critic for continuous action spaces
//  Based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
//

import Collections
import Foundation
import MLX
import MLXNN
import MLXOptimizers

public class TrainableParameter: Module {
    @ParameterInfo(key: "value") var value: MLXArray
    
    public init(_ initialValue: MLXArray) {
        self._value.wrappedValue = initialValue
        super.init()
    }
    
    public init(_ initialValue: Float) {
        self._value.wrappedValue = MLXArray(initialValue)
        super.init()
    }
    
    public var item: Float {
        value.item(Float.self)
    }
}

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

        let logProbNorm = -0.5 * (pow((x_t - mean) / std, 2.0) + 2.0 * logStd + logPiConstant)
        let logProbCorrection = log(1.0 - pow(y_t, 2.0) + epsilon)
        let logProb = (logProbNorm - logProbCorrection).sum(axis: -1, keepDims: true)
        
        return (action, logProb, mean)
    }
    
    public func getDeterministicAction(obs: MLXArray) -> MLXArray {
        let (mean, _) = self(obs)
        return tanh(mean) * actionScale + actionBias
    }
}

public struct SACExperience {
    public let observation: MLXArray
    public let nextObservation: MLXArray
    public let action: MLXArray
    public let reward: MLXArray
    public let terminated: MLXArray
}

public class SACReplayBuffer {
    public let capacity: Int
    public let stateSize: Int
    public let actionSize: Int
    
    var obsBuffer: MLXArray
    var nextObsBuffer: MLXArray
    var actionBuffer: MLXArray
    var rewardBuffer: MLXArray
    var terminatedBuffer: MLXArray
    
    var ptr: Int = 0
    var size: Int = 0
    
    public init(capacity: Int, stateSize: Int, actionSize: Int) {
        self.capacity = capacity
        self.stateSize = stateSize
        self.actionSize = actionSize
        
        self.obsBuffer = MLXArray.zeros([capacity, stateSize])
        self.nextObsBuffer = MLXArray.zeros([capacity, stateSize])
        self.actionBuffer = MLXArray.zeros([capacity, actionSize])
        self.rewardBuffer = MLXArray.zeros([capacity, 1])
        self.terminatedBuffer = MLXArray.zeros([capacity, 1])
    }
    
    public func push(_ experience: SACExperience) {
        let obs = experience.observation.ndim == 1 ? experience.observation : experience.observation.reshaped([stateSize])
        let nextObs = experience.nextObservation.ndim == 1 ? experience.nextObservation : experience.nextObservation.reshaped([stateSize])
        let action = experience.action.ndim == 1 ? experience.action : experience.action.reshaped([actionSize])
        
        obsBuffer[ptr] = obs
        nextObsBuffer[ptr] = nextObs
        actionBuffer[ptr] = action
        rewardBuffer[ptr] = experience.reward.reshaped([1])
        terminatedBuffer[ptr] = experience.terminated.reshaped([1])
        
        ptr = (ptr + 1) % capacity
        size = min(size + 1, capacity)
    }
    
    public func sample(batchSize: Int) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let safeBatchSize = min(batchSize, size)
        
        let indices = MLX.randInt(low: 0, high: size, [safeBatchSize])
        
        let bObs = obsBuffer[indices]
        let bNextObs = nextObsBuffer[indices]
        let bActions = actionBuffer[indices]
        let bRewards = rewardBuffer[indices]
        let bTerminated = terminatedBuffer[indices]
        
        return (bObs, bNextObs, bActions, bRewards, bTerminated)
    }
    
    public var count: Int { size }
}

public class SACAgent: ContinuousDeepRLAgent {
    public let actor: SACActorNetwork
    public let qf1: SoftQNetwork
    public let qf2: SoftQNetwork
    public let qf1Target: SoftQNetwork
    public let qf2Target: SoftQNetwork
    
    public let actorOptimizer: AdamW
    public let qOptimizer: AdamW
    
    public let memory: SACReplayBuffer
    
    public let stateSize: Int
    public let actionSize: Int
    public let gamma: Float
    public let tau: Float
    public let batchSize: Int
    public var alpha: Float
    public let autotune: Bool
    public var targetEntropy: Float
    public var targetEntropyArray: MLXArray
    public let logAlphaModule: TrainableParameter?
    public let alphaOptimizer: AdamW?
    
    public var steps: Int = 0
    
    private let tauArray: MLXArray
    private let oneMinusTauArray: MLXArray
    
    public init(
        observationSpace: Box,
        actionSpace: Box,
        hiddenSize: Int = 256,
        learningRate: Float = 3e-4,
        gamma: Float = 0.99,
        tau: Float = 0.005,
        alpha: Float = 0.2,
        autotune: Bool = true,
        batchSize: Int = 256,
        bufferSize: Int = 100000
    ) {
        let obsShape = observationSpace.shape ?? observationSpace.low.shape
        self.stateSize = obsShape.reduce(1, *)
        
        let actShape = actionSpace.shape ?? actionSpace.low.shape
        self.actionSize = actShape.reduce(1, *)
        
        let actionLow = actionSpace.low[0].item(Float.self)
        let actionHigh = actionSpace.high[0].item(Float.self)
        
        self.actor = SACActorNetwork(
            numObservations: stateSize,
            numActions: actionSize,
            hiddenSize: hiddenSize,
            actionSpaceLow: actionLow,
            actionSpaceHigh: actionHigh
        )
        
        self.qf1 = SoftQNetwork(numObservations: stateSize, numActions: actionSize, hiddenSize: hiddenSize)
        self.qf2 = SoftQNetwork(numObservations: stateSize, numActions: actionSize, hiddenSize: hiddenSize)
        self.qf1Target = SoftQNetwork(numObservations: stateSize, numActions: actionSize, hiddenSize: hiddenSize)
        self.qf2Target = SoftQNetwork(numObservations: stateSize, numActions: actionSize, hiddenSize: hiddenSize)
        
        self.qf1Target.update(parameters: qf1.parameters())
        self.qf2Target.update(parameters: qf2.parameters())
        
        self.actorOptimizer = AdamW(learningRate: learningRate)
        self.qOptimizer = AdamW(learningRate: learningRate)
        
        self.memory = SACReplayBuffer(capacity: bufferSize, stateSize: stateSize, actionSize: actionSize)
        
        self.gamma = gamma
        self.tau = tau
        self.batchSize = batchSize
        self.alpha = alpha
        self.autotune = autotune
        
        if autotune {
            self.targetEntropy = -Float(actionSize)
            self.targetEntropyArray = MLXArray(-Float(actionSize))
            self.logAlphaModule = TrainableParameter(log(alpha))
            self.alphaOptimizer = AdamW(learningRate: learningRate)
        } else {
            self.targetEntropy = 0
            self.targetEntropyArray = MLXArray(0.0)
            self.logAlphaModule = nil
            self.alphaOptimizer = nil
        }
        
        self.tauArray = MLXArray(tau)
        self.oneMinusTauArray = MLXArray(1.0 - tau)
        
        eval(actor, qf1, qf2, qf1Target, qf2Target)
        if let logAlphaModule {
            eval(logAlphaModule)
        }
    }
    
    public func chooseAction(state: MLXArray, key: inout MLXArray, deterministic: Bool = false) -> MLXArray {
        let stateRow = state.count == stateSize ? state.reshaped([1, stateSize]) : state
        
        if deterministic {
            return actor.getDeterministicAction(obs: stateRow).reshaped([actionSize])
        }
        
        let (k1, k2) = MLX.split(key: key)
        key = k2
        let (action, _, _) = actor.sample(obs: stateRow, key: k1)
        return action.reshaped([actionSize])
    }
    
    public func store(state: MLXArray, action: MLXArray, reward: Float, nextState: MLXArray, terminated: Bool) {
        eval(state, action, nextState)
        
        let exp = SACExperience(
            observation: state,
            nextObservation: nextState,
            action: action,
            reward: MLXArray(reward),
            terminated: MLXArray(terminated ? 1.0 : 0.0)
        )
        memory.push(exp)
    }
    
    private func updateQ(
        batchObs: MLXArray,
        batchNextObs: MLXArray,
        batchActions: MLXArray,
        batchRewards: MLXArray,
        batchTerminated: MLXArray,
        rngKey: MLXArray,
        alphaVal: MLXArray
    ) -> MLXArray {
        let gammaArr = MLXArray(gamma)
        
        let (nextActions, nextLogProbs, _) = actor.sample(obs: batchNextObs, key: rngKey)
        let qf1NextTarget = qf1Target.callAsFunction(obs: batchNextObs, action: nextActions)
        let qf2NextTarget = qf2Target.callAsFunction(obs: batchNextObs, action: nextActions)
        let minQfNextTarget = minimum(qf1NextTarget, qf2NextTarget) - alphaVal * nextLogProbs
        let nextQValue = stopGradient(batchRewards + (1.0 - batchTerminated) * gammaArr * minQfNextTarget)
        
        let qf1LossAndGrad = valueAndGrad(model: qf1) { (model: SoftQNetwork, obs: MLXArray, targets: MLXArray) -> MLXArray in
            let qVal = model.callAsFunction(obs: obs, action: batchActions)
            return pow(qVal - targets, 2.0).mean()
        }
        let (qf1LossValue, qf1Grads) = qf1LossAndGrad(qf1, batchObs, nextQValue)
        qOptimizer.update(model: qf1, gradients: qf1Grads)
        
        let qf2LossAndGrad = valueAndGrad(model: qf2) { (model: SoftQNetwork, obs: MLXArray, targets: MLXArray) -> MLXArray in
            let qVal = model.callAsFunction(obs: obs, action: batchActions)
            return pow(qVal - targets, 2.0).mean()
        }
        let (qf2LossValue, qf2Grads) = qf2LossAndGrad(qf2, batchObs, nextQValue)
        qOptimizer.update(model: qf2, gradients: qf2Grads)
        
        return qf1LossValue + qf2LossValue
    }
    
    private func updateActor(
        batchObs: MLXArray,
        rngKey: MLXArray,
        alphaVal: MLXArray
    ) -> (actorLoss: MLXArray, meanLogPi: MLXArray) {
        let (_, logPiForAlpha, _) = actor.sample(obs: batchObs, key: rngKey)
        let meanLogPi = logPiForAlpha.mean()
        
        let actorLossAndGrad = valueAndGrad(model: actor) { [self] (model: SACActorNetwork, obs: MLXArray, key: MLXArray) -> MLXArray in
            let (piAct, logP, _) = model.sample(obs: obs, key: key)
            let q1Pi = qf1.callAsFunction(obs: obs, action: piAct)
            let q2Pi = qf2.callAsFunction(obs: obs, action: piAct)
            let minQ = minimum(q1Pi, q2Pi)
            return (alphaVal * logP - minQ).mean()
        }
        
        let (actorLossValue, actorGrads) = actorLossAndGrad(actor, batchObs, rngKey)
        actorOptimizer.update(model: actor, gradients: actorGrads)
        
        return (actorLossValue, meanLogPi)
    }
    
    public func update() -> (qLoss: Float, actorLoss: Float, alphaLoss: Float)? {
        guard memory.count >= batchSize else { return nil }
        
        steps += 1
        
        let (batchObs, batchNextObs, batchActions, batchRewards, batchTerminated) = memory.sample(batchSize: batchSize)
        
        let batchRewardsFloat = batchRewards.asType(.float32)
        let batchTerminatedFloat = batchTerminated.asType(.float32)
        
        let key = MLX.key(UInt64(steps))
        let (k1, k2) = MLX.split(key: key)
        
        let logAlpha = logAlphaModule?.value ?? MLXArray(log(alpha))
        let alphaVal = exp(logAlpha)
        
        let totalQLoss = updateQ(
            batchObs: batchObs,
            batchNextObs: batchNextObs,
            batchActions: batchActions,
            batchRewards: batchRewardsFloat,
            batchTerminated: batchTerminatedFloat,
            rngKey: k1,
            alphaVal: alphaVal
        )
        
        let (actorLossValue, meanLogPi) = updateActor(
            batchObs: batchObs,
            rngKey: k2,
            alphaVal: alphaVal
        )
        
        softUpdateTargetNetworks()
        
        var alphaLossArray: MLXArray = MLXArray(0.0)
        
        if autotune, let logAlphaModule, let alphaOptimizer {
            // Compute alpha loss using valueAndGrad with the trainable module
            let alphaLossAndGrad = valueAndGrad(model: logAlphaModule) { (module: TrainableParameter, meanLogPiVal: MLXArray, targetEnt: MLXArray) -> MLXArray in
                return -module.value * stopGradient(meanLogPiVal + targetEnt)
            }
            
            let (alphaLoss, alphaGrads) = alphaLossAndGrad(logAlphaModule, meanLogPi, targetEntropyArray)
            alphaLossArray = alphaLoss
            
            alphaOptimizer.update(model: logAlphaModule, gradients: alphaGrads)
            
            eval(
                totalQLoss, actorLossValue, alphaLossArray,
                actor.parameters(), qf1.parameters(), qf2.parameters(),
                qf1Target.parameters(), qf2Target.parameters(),
                logAlphaModule.parameters()
            )
            
            // Clamp log_alpha to prevent alpha from becoming too small (policy too deterministic)
            // or too large (too much exploration). 
            let minLogAlpha: Float = -3.0
            let maxLogAlpha: Float = -0.7
            let clampedLogAlpha = min(max(logAlphaModule.item, minLogAlpha), maxLogAlpha)
            logAlphaModule.value = MLXArray(clampedLogAlpha)
            alpha = exp(clampedLogAlpha)
            
            return (totalQLoss.item(Float.self), actorLossValue.item(Float.self), alphaLossArray.item(Float.self))
        }
        
        eval(
            totalQLoss, actorLossValue,
            actor.parameters(), qf1.parameters(), qf2.parameters(),
            qf1Target.parameters(), qf2Target.parameters()
        )
        
        return (totalQLoss.item(Float.self), actorLossValue.item(Float.self), 0.0)
    }
    
    private func softUpdateTargetNetworks() {
        func softUpdate(target: Module, source: Module) {
            let sourceParams = source.parameters().flattened()
            let targetParams = target.parameters().flattened()
            let sourceDict = Dictionary(uniqueKeysWithValues: sourceParams)
            
            var updated = [(String, MLXArray)]()
            updated.reserveCapacity(targetParams.count)
            
            for (key, targetParam) in targetParams {
                if let sourceParam = sourceDict[key] {
                    updated.append((key, oneMinusTauArray * targetParam + tauArray * sourceParam))
                }
            }
            
            let newParams = NestedDictionary<String, MLXArray>.unflattened(updated)
            target.update(parameters: newParams)
        }
        
        softUpdate(target: qf1Target, source: qf1)
        softUpdate(target: qf2Target, source: qf2)
    }
}

public class SACAgentVmap: ContinuousDeepRLAgent {
    public let actor: SACActorNetwork
    public let qEnsemble: EnsembleQNetwork
    public let qEnsembleTarget: EnsembleQNetwork
    
    public let actorOptimizer: AdamW
    public let qOptimizer: AdamW
    
    public let memory: SACReplayBuffer
    
    public let stateSize: Int
    public let actionSize: Int
    public let gamma: Float
    public let tau: Float
    public let batchSize: Int
    public var alpha: Float
    public let autotune: Bool
    public var targetEntropy: Float
    public var targetEntropyArray: MLXArray
    public let logAlphaModule: TrainableParameter?
    public let alphaOptimizer: AdamW?
    
    public var steps: Int = 0
    
    private let tauArray: MLXArray
    private let oneMinusTauArray: MLXArray
    private let gammaArray: MLXArray
    
    public init(
        observationSpace: Box,
        actionSpace: Box,
        hiddenSize: Int = 256,
        learningRate: Float = 3e-4,
        gamma: Float = 0.99,
        tau: Float = 0.005,
        alpha: Float = 0.2,
        autotune: Bool = true,
        batchSize: Int = 256,
        bufferSize: Int = 100000
    ) {
        let obsShape = observationSpace.shape ?? observationSpace.low.shape
        self.stateSize = obsShape.reduce(1, *)
        
        let actShape = actionSpace.shape ?? actionSpace.low.shape
        self.actionSize = actShape.reduce(1, *)
        
        let actionLow = actionSpace.low[0].item(Float.self)
        let actionHigh = actionSpace.high[0].item(Float.self)
        
        self.actor = SACActorNetwork(
            numObservations: stateSize,
            numActions: actionSize,
            hiddenSize: hiddenSize,
            actionSpaceLow: actionLow,
            actionSpaceHigh: actionHigh
        )
        
        self.qEnsemble = EnsembleQNetwork(
            numObservations: stateSize,
            numActions: actionSize,
            numEnsemble: 2,
            hiddenSize: hiddenSize
        )
        self.qEnsembleTarget = EnsembleQNetwork(
            numObservations: stateSize,
            numActions: actionSize,
            numEnsemble: 2,
            hiddenSize: hiddenSize
        )
        
        self.qEnsembleTarget.update(parameters: qEnsemble.parameters())
        
        self.actorOptimizer = AdamW(learningRate: learningRate)
        self.qOptimizer = AdamW(learningRate: learningRate)
        
        self.memory = SACReplayBuffer(capacity: bufferSize, stateSize: stateSize, actionSize: actionSize)
        
        self.gamma = gamma
        self.tau = tau
        self.batchSize = batchSize
        self.alpha = alpha
        self.autotune = autotune
        
        if autotune {
            self.targetEntropy = -Float(actionSize)
            self.targetEntropyArray = MLXArray(-Float(actionSize))
            self.logAlphaModule = TrainableParameter(log(alpha))
            self.alphaOptimizer = AdamW(learningRate: learningRate)
        } else {
            self.targetEntropy = 0
            self.targetEntropyArray = MLXArray(0.0)
            self.logAlphaModule = nil
            self.alphaOptimizer = nil
        }
        
        self.tauArray = MLXArray(tau)
        self.oneMinusTauArray = MLXArray(1.0 - tau)
        self.gammaArray = MLXArray(gamma)
        
        eval(actor, qEnsemble, qEnsembleTarget)
        if let logAlphaModule {
            eval(logAlphaModule)
        }
    }
    
    public func chooseAction(state: MLXArray, key: inout MLXArray, deterministic: Bool = false) -> MLXArray {
        let stateRow = state.count == stateSize ? state.reshaped([1, stateSize]) : state
        
        if deterministic {
            return actor.getDeterministicAction(obs: stateRow).reshaped([actionSize])
        }
        
        let (k1, k2) = MLX.split(key: key)
        key = k2
        let (action, _, _) = actor.sample(obs: stateRow, key: k1)
        return action.reshaped([actionSize])
    }
    
    public func store(state: MLXArray, action: MLXArray, reward: Float, nextState: MLXArray, terminated: Bool) {
        eval(state, action, nextState)
        
        let exp = SACExperience(
            observation: state,
            nextObservation: nextState,
            action: action,
            reward: MLXArray(reward),
            terminated: MLXArray(terminated ? 1.0 : 0.0)
        )
        memory.push(exp)
    }
    
    private func updateQ(
        batchObs: MLXArray,
        batchNextObs: MLXArray,
        batchActions: MLXArray,
        batchRewards: MLXArray,
        batchTerminated: MLXArray,
        rngKey: MLXArray,
        alphaVal: MLXArray
    ) -> MLXArray {
        let (nextActions, nextLogProbs, _) = actor.sample(obs: batchNextObs, key: rngKey)
        
        let minQfNextTarget = qEnsembleTarget.minQ(obs: batchNextObs, action: nextActions)
        let targetWithEntropy = minQfNextTarget - alphaVal * nextLogProbs
        let nextQValue = stopGradient(batchRewards + (1.0 - batchTerminated) * gammaArray * targetWithEntropy)
        
        let qLossAndGrad = valueAndGrad(model: qEnsemble) { (model: EnsembleQNetwork, obs: MLXArray, targets: MLXArray) -> MLXArray in
            let allQ = model.callAsFunction(obs: obs, action: batchActions)
            let q1 = allQ[0]
            let q2 = allQ[1]
            let loss1 = pow(q1 - targets, 2.0).mean()
            let loss2 = pow(q2 - targets, 2.0).mean()
            return loss1 + loss2
        }
        
        let (totalQLoss, qGrads) = qLossAndGrad(qEnsemble, batchObs, nextQValue)
        qOptimizer.update(model: qEnsemble, gradients: qGrads)
        
        return totalQLoss
    }
    
    private func updateActor(
        batchObs: MLXArray,
        rngKey: MLXArray,
        alphaVal: MLXArray
    ) -> (actorLoss: MLXArray, meanLogPi: MLXArray) {
        let (_, logPiForAlpha, _) = actor.sample(obs: batchObs, key: rngKey)
        let meanLogPi = logPiForAlpha.mean()
        
        let actorLossAndGrad = valueAndGrad(model: actor) { [self] (model: SACActorNetwork, obs: MLXArray, key: MLXArray) -> MLXArray in
            let (piAct, logP, _) = model.sample(obs: obs, key: key)
            let minQ = qEnsemble.minQ(obs: obs, action: piAct)
            return (alphaVal * logP - minQ).mean()
        }
        
        let (actorLossValue, actorGrads) = actorLossAndGrad(actor, batchObs, rngKey)
        actorOptimizer.update(model: actor, gradients: actorGrads)
        
        return (actorLossValue, meanLogPi)
    }
    
    public func update() -> (qLoss: Float, actorLoss: Float, alphaLoss: Float)? {
        guard memory.count >= batchSize else { return nil }
        
        steps += 1
        
        let (batchObs, batchNextObs, batchActions, batchRewards, batchTerminated) = memory.sample(batchSize: batchSize)
        
        let batchRewardsFloat = batchRewards.asType(.float32)
        let batchTerminatedFloat = batchTerminated.asType(.float32)
        
        let key = MLX.key(UInt64(steps))
        let (k1, k2) = MLX.split(key: key)
        
        let logAlpha = logAlphaModule?.value ?? MLXArray(log(alpha))
        let alphaVal = exp(logAlpha)
        
        let totalQLoss = updateQ(
            batchObs: batchObs,
            batchNextObs: batchNextObs,
            batchActions: batchActions,
            batchRewards: batchRewardsFloat,
            batchTerminated: batchTerminatedFloat,
            rngKey: k1,
            alphaVal: alphaVal
        )
        
        let (actorLossValue, meanLogPi) = updateActor(
            batchObs: batchObs,
            rngKey: k2,
            alphaVal: alphaVal
        )
        
        softUpdateTargetNetwork()
        
        var alphaLossArray: MLXArray = MLXArray(0.0)
        
        if autotune, let logAlphaModule, let alphaOptimizer {
            let alphaLossAndGrad = valueAndGrad(model: logAlphaModule) { (module: TrainableParameter, meanLogPiVal: MLXArray, targetEnt: MLXArray) -> MLXArray in
                return -module.value * stopGradient(meanLogPiVal + targetEnt)
            }
            
            let (alphaLoss, alphaGrads) = alphaLossAndGrad(logAlphaModule, meanLogPi, targetEntropyArray)
            alphaLossArray = alphaLoss
            
            alphaOptimizer.update(model: logAlphaModule, gradients: alphaGrads)
            
            eval(
                totalQLoss, actorLossValue, alphaLossArray,
                actor.parameters(), qEnsemble.parameters(), qEnsembleTarget.parameters(),
                logAlphaModule.parameters()
            )
            
            let minLogAlpha: Float = -3.0
            let maxLogAlpha: Float = -0.7
            let clampedLogAlpha = min(max(logAlphaModule.item, minLogAlpha), maxLogAlpha)
            logAlphaModule.value = MLXArray(clampedLogAlpha)
            alpha = exp(clampedLogAlpha)
            
            return (totalQLoss.item(Float.self), actorLossValue.item(Float.self), alphaLossArray.item(Float.self))
        }
        
        eval(
            totalQLoss, actorLossValue,
            actor.parameters(), qEnsemble.parameters(), qEnsembleTarget.parameters()
        )
        
        return (totalQLoss.item(Float.self), actorLossValue.item(Float.self), 0.0)
    }
    
    private func softUpdateTargetNetwork() {
        let sourceParams = qEnsemble.parameters().flattened()
        let targetParams = qEnsembleTarget.parameters().flattened()
        let sourceDict = Dictionary(uniqueKeysWithValues: sourceParams)
        
        var updated = [(String, MLXArray)]()
        updated.reserveCapacity(targetParams.count)
        
        for (key, targetParam) in targetParams {
            if let sourceParam = sourceDict[key] {
                updated.append((key, oneMinusTauArray * targetParam + tauArray * sourceParam))
            }
        }
        
        let newParams = NestedDictionary<String, MLXArray>.unflattened(updated)
        qEnsembleTarget.update(parameters: newParams)
    }
}
