//
//  DQN.swift
//

import Collections
import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// struct representing a single 'experience' in the environment
struct Experience {
    let observation: MLXArray
    let nextObservation: MLXArray
    let action: MLXArray
    let reward: MLXArray
    let terminated: MLXArray

    init(
        observation: MLXArray,
        nextObservation: MLXArray,
        action: MLXArray,
        reward: MLXArray,
        terminated: MLXArray
    ) {
        self.observation = observation
        self.nextObservation = nextObservation
        self.action = action
        self.reward = reward
        self.terminated = terminated
    }
}

public class ReplayMemory {
    let capacity: Int
    let stateSize: Int
    let actionSize: Int = 1
    
    var obsBuffer: [Float]
    var nextObsBuffer: [Float]
    var actionBuffer: [Int32]
    var rewardBuffer: [Float]
    var terminatedBuffer: [Float]
    
    var ptr: Int = 0
    var size: Int = 0
    
    init(capacity: Int, stateSize: Int) {
        self.capacity = capacity
        self.stateSize = stateSize
        
        // Pre-allocate memory with repeating values
        self.obsBuffer = [Float](repeating: 0, count: capacity * stateSize)
        self.nextObsBuffer = [Float](repeating: 0, count: capacity * stateSize)
        self.actionBuffer = [Int32](repeating: 0, count: capacity)
        self.rewardBuffer = [Float](repeating: 0, count: capacity)
        self.terminatedBuffer = [Float](repeating: 0, count: capacity)
    }
    
    func push(_ experience: Experience) {
        let obsFlat = experience.observation.asArray(Float.self)
        let nextObsFlat = experience.nextObservation.asArray(Float.self)
        let actionScalar = experience.action.item(Int32.self)
        let rewardScalar = experience.reward.item(Float.self)
        let termScalar = experience.terminated.item(Float.self)
        
        let startIdx = ptr * stateSize
        
        for i in 0..<stateSize {
            obsBuffer[startIdx + i] = obsFlat[i]
            nextObsBuffer[startIdx + i] = nextObsFlat[i]
        }
        
        actionBuffer[ptr] = actionScalar
        rewardBuffer[ptr] = rewardScalar
        terminatedBuffer[ptr] = termScalar
        
        ptr = (ptr + 1) % capacity
        size = min(size + 1, capacity)
    }
    
    func sample(batchSize: Int) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let safeBatchSize = min(batchSize, size)
        
        var indices = [Int]()
        indices.reserveCapacity(safeBatchSize)
        for _ in 0..<safeBatchSize {
            indices.append(Int.random(in: 0..<size))
        }
        
        var bObs = [Float]()
        var bNextObs = [Float]()
        var bActions = [Int32]()
        var bRewards = [Float]()
        var bTerminated = [Float]()
        
        bObs.reserveCapacity(safeBatchSize * stateSize)
        bNextObs.reserveCapacity(safeBatchSize * stateSize)
        bActions.reserveCapacity(safeBatchSize)
        bRewards.reserveCapacity(safeBatchSize)
        bTerminated.reserveCapacity(safeBatchSize)
        
        for idx in indices {
            let start = idx * stateSize

            bObs.append(contentsOf: obsBuffer[start..<(start+stateSize)])
            bNextObs.append(contentsOf: nextObsBuffer[start..<(start+stateSize)])
            bActions.append(actionBuffer[idx])
            bRewards.append(rewardBuffer[idx])
            bTerminated.append(terminatedBuffer[idx])
        }
        
        let mlxObs = MLXArray(bObs).reshaped([safeBatchSize, stateSize])
        let mlxNextObs = MLXArray(bNextObs).reshaped([safeBatchSize, stateSize])
        let mlxActions = MLXArray(bActions).reshaped([safeBatchSize, 1])
        let mlxRewards = MLXArray(bRewards).reshaped([safeBatchSize, 1])
        let mlxTerminated = MLXArray(bTerminated).reshaped([safeBatchSize, 1])
        
        return (mlxObs, mlxNextObs, mlxActions, mlxRewards, mlxTerminated)
    }
}

/// Creates a Linear layer with Xavier/Glorot uniform initialization
private func xavierLinear(_ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true) -> Linear {
    // xavier uniform initialization: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
    let bound = sqrt(6.0 / Float(inputDimensions + outputDimensions))
    let weight = MLX.uniform(
        low: -bound,
        high: bound,
        [outputDimensions, inputDimensions]
    )
    let biasArray: MLXArray? = bias ? MLX.zeros([outputDimensions]) : nil
    return Linear(weight: weight, bias: biasArray)
}

public class QNetwork: Module {
    @ModuleInfo var layer1: Linear
    @ModuleInfo var layer2: Linear
    @ModuleInfo var layer3: Linear

    public init(
        numObservations: Int,
        numActions: Int,
        hiddenSize: Int = 128
    ) {
        self._layer1.wrappedValue = xavierLinear(numObservations, hiddenSize)
        self._layer2.wrappedValue = xavierLinear(hiddenSize, hiddenSize)
        self._layer3.wrappedValue = xavierLinear(hiddenSize, numActions)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        x = relu(layer1(x))
        x = relu(layer2(x))
        return layer3(x)
    }
}

public enum TargetUpdateStrategy {
    /// Soft update: θ′ ← τ θ + (1 −τ )θ′ every step
    case soft(tau: Float)
    /// Hard update: θ′ ← θ every N steps
    case hard(frequency: Int)
}

public class DQNAgent: DeepRLAgent {
    public let batchSize: Int
    public let memory: ReplayMemory

    public var stateSize: Int
    public var actionSize: Int

    public var gamma: Float
    public var epsilon: Float
    public var epsilonStart: Float
    public var epsilonEnd: Float
    public var epsilonDecaySteps: Int
    public var targetUpdateStrategy: TargetUpdateStrategy
    public var learningRate: Float
    public var gradClipNorm: Float

    public var optim: AdamW

    public let policyNetwork: QNetwork
    public let targetNetwork: QNetwork

    public var steps: Int
    private var explorationSteps: Int = 0
    public var episodeDurations: [Int] = []
    

    private let gammaArray: MLXArray
    private let tauArray: MLXArray?
    private let oneMinusTauArray: MLXArray?

    public convenience init(
        observationSpace: Box,
        actionSpace: Discrete,
        hiddenDimensions: Int,
        learningRate: Float,
        gamma: Float,
        epsilon: Float,
        epsilonEnd: Float,
        epsilonDecaySteps: Int,
        tau: Float,
        batchSize: Int,
        bufferSize: Int,
        gradClipNorm: Float,
        targetUpdateStrategy: TargetUpdateStrategy? = nil
    ) {
        let obsSize = (observationSpace.shape ?? observationSpace.low.shape)
            .reduce(1, *)
        self.init(
            batchSize: batchSize,
            stateSize: obsSize,
            actionSize: actionSpace.n,
            gamma: gamma,
            epsilonStart: epsilon,
            epsilonEnd: epsilonEnd,
            epsilonDecaySteps: epsilonDecaySteps,
            targetUpdateStrategy: targetUpdateStrategy ?? .soft(tau: tau),
            learningRate: learningRate,
            optim: AdamW(learningRate: learningRate),
            gradClipNorm: gradClipNorm,
            bufferCapacity: bufferSize
        )
    }

    init(
        batchSize: Int,
        stateSize: Int,
        actionSize: Int,
        gamma: Float,
        epsilonStart: Float,
        epsilonEnd: Float,
        epsilonDecaySteps: Int,
        targetUpdateStrategy: TargetUpdateStrategy = .soft(tau: 0.005),
        learningRate: Float,
        optim: AdamW,
        gradClipNorm: Float = 100.0,
        bufferCapacity: Int = 10000
    ) {
        self.batchSize = batchSize
        self.memory = ReplayMemory(capacity: bufferCapacity, stateSize: stateSize)

        self.stateSize = stateSize
        self.actionSize = actionSize

        self.gamma = gamma
        self.epsilon = epsilonStart
        self.epsilonStart = epsilonStart
        self.epsilonEnd = epsilonEnd
        self.epsilonDecaySteps = max(1, epsilonDecaySteps)
        self.targetUpdateStrategy = targetUpdateStrategy
        self.learningRate = learningRate
        self.gradClipNorm = gradClipNorm

        self.optim = AdamW(learningRate: learningRate)

        self.policyNetwork = QNetwork(
            numObservations: stateSize,
            numActions: actionSize
        )
        self.targetNetwork = QNetwork(
            numObservations: stateSize,
            numActions: actionSize
        )
        self.targetNetwork.update(parameters: policyNetwork.parameters())
        
        self.gammaArray = MLXArray(gamma)
        if case .soft(let tau) = targetUpdateStrategy {
            self.tauArray = MLXArray(tau)
            self.oneMinusTauArray = MLXArray(1.0 - tau)
        } else {
            self.tauArray = nil
            self.oneMinusTauArray = nil
        }
        
        eval(self.policyNetwork)
        eval(self.targetNetwork)

        self.steps = 0
    }

    public func chooseAction(
        state: MLXArray,
        actionSpace: Discrete,
        key: inout MLXArray
    ) -> MLXArray {
        explorationSteps += 1
        updateEpsilonSchedule()
        
        if Float.random(in: 0..<1) < epsilon {
            let randomAction = Int32.random(in: 0..<Int32(actionSize))
            return MLXArray([randomAction]).reshaped([1, 1])
        } else {
            let stateRow = state.ndim == 1 ? state.reshaped([1, stateSize]) : state
            let qValues = policyNetwork(stateRow)
            eval(qValues)
            let actionIndex = argMax(qValues, axis: 1)
            return actionIndex.reshaped([1, 1])
        }
    }

    public func store(
        state: MLXArray,
        action: MLXArray,
        reward: Float,
        nextState: MLXArray,
        terminated: Bool
    ) {
        let e = Experience(
            observation: state,
            nextObservation: nextState,
            action: action,
            reward: MLXArray(reward),
            terminated: MLXArray(terminated ? 1.0 : 0.0)
        )
        self.memory.push(e)
    }

    private func trainStep(
        batchObs: MLXArray,
        batchNextObs: MLXArray,
        batchActions: MLXArray,
        batchRewards: MLXArray,
        batchTerminated: MLXArray
    ) -> (loss: MLXArray, gradNorm: MLXArray, meanQ: MLXArray, tdError: MLXArray) {
        let nextQValues = targetNetwork(batchNextObs)
        let maxNextQ = nextQValues.max(axis: 1).reshaped([-1, 1])
        
        // TD target: r + γ * max_a' Q_target(s', a') * (1 - done)
        let targetQValues = stopGradient(
            batchRewards + (gammaArray * maxNextQ * (1 - batchTerminated))
        )
        
        let lossAndGrad = valueAndGrad(model: policyNetwork) { (model: QNetwork, obs: MLXArray, targets: MLXArray) -> MLXArray in
            let predictedQ = model(obs)
            let selectedQ = takeAlong(predictedQ, batchActions, axis: 1)
            return smoothL1Loss(predictions: selectedQ, targets: targets, reduction: .mean)
        }
        
        let (lossValue, grads) = lossAndGrad(policyNetwork, batchObs, targetQValues)
        let (clippedGrads, gradNormValue) = clipGradNorm(gradients: grads, maxNorm: gradClipNorm)
        optim.update(model: policyNetwork, gradients: clippedGrads)
        
        let currentQValues = policyNetwork(batchObs)
        let meanQValue = currentQValues.max(axis: 1).mean()
        let selectedCurrentQ = takeAlong(currentQValues, batchActions, axis: 1)
        let tdError = abs(selectedCurrentQ - targetQValues).mean()
        
        return (lossValue, gradNormValue, meanQValue, tdError)
    }
    
    public func update() -> (
        loss: Float, meanQ: Float, gradNorm: Float, tdError: Float
    )? {
        guard memory.size >= batchSize else { return nil }
        
        steps += 1
        
        let (batchObs, batchNextObs, batchActions, batchRewards, batchTerminated) = memory.sample(batchSize: batchSize)
        
        let (lossValue, gradNormValue, meanQArray, tdErrorArray) = trainStep(
            batchObs: batchObs,
            batchNextObs: batchNextObs,
            batchActions: batchActions,
            batchRewards: batchRewards,
            batchTerminated: batchTerminated
        )
        
        updateTargetNetwork()
        
        eval(
            lossValue, gradNormValue, meanQArray, tdErrorArray,
            policyNetwork.parameters(), targetNetwork.parameters()
        )
        
        return (
            lossValue.item(Float.self),
            meanQArray.item(Float.self),
            gradNormValue.item(Float.self),
            tdErrorArray.item(Float.self)
        )
    }
        
    private func updateTargetNetwork() {
        switch targetUpdateStrategy {
        case .soft:
            softUpdate(target: targetNetwork, source: policyNetwork)
        case .hard(let frequency):
            if steps % frequency == 0 {
                hardUpdate(target: targetNetwork, source: policyNetwork)
            }
        }
    }
    
    private func hardUpdate(target: Module, source: Module) {
        target.update(parameters: source.parameters())
    }

    /// θ′ ← τ θ + (1 −τ )θ′
    private func softUpdate(target: Module, source: Module) {
        guard let tauArray = tauArray, let oneMinusTauArray = oneMinusTauArray else { return }
        
        let sourceParams = source.parameters().flattened()
        let targetParams = target.parameters().flattened()
        let sourceDict = Dictionary(uniqueKeysWithValues: sourceParams)
        
        var updatedParams = [(String, MLXArray)]()
        updatedParams.reserveCapacity(targetParams.count)
        
        for (key, targetParam) in targetParams {
            if let sourceParam = sourceDict[key] {
                let updated = oneMinusTauArray * targetParam + tauArray * sourceParam
                updatedParams.append((key, updated))
            }
        }
        
        let newParams = NestedDictionary<String, MLXArray>.unflattened(updatedParams)
        target.update(parameters: newParams)
    }

    private func updateEpsilonSchedule() {
        guard epsilonStart != epsilonEnd else {
            epsilon = epsilonEnd
            return
        }
        
        // Linear decay
        let progress = min(Float(explorationSteps) / Float(epsilonDecaySteps), 1.0)
        epsilon = epsilonStart + (epsilonEnd - epsilonStart) * progress
    }
    
    /// Get the current exploration step count (for saving)
    public var currentExplorationSteps: Int {
        return explorationSteps
    }
    
    /// Restore the exploration step count
    public func setExplorationSteps(_ steps: Int) {
        self.explorationSteps = steps
        updateEpsilonSchedule()
    }
    
    /// Get current training step count
    public var currentSteps: Int {
        return steps
    }
    
    /// Set training step count
    public func setSteps(_ newSteps: Int) {
        self.steps = newSteps
    }
}
