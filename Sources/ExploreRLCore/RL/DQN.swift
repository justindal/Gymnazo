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

/// DQN Replay buffer for storing experiences
public class ReplayMemory {
    let capacity: Int
    var memory: Deque<Experience>

    init(capacity: Int) {
        self.capacity = capacity
        self.memory = Deque()
        self.memory.reserveCapacity(capacity)
    }

    func push(_ experience: Experience) {
        if memory.count >= capacity {
            memory.removeFirst()
        }
        self.memory.append(experience)
    }

    func sample(batchSize: Int) -> [Experience] {
        precondition(
            !self.memory.isEmpty,
            "Cannot sample from an empty replay memory."
        )

        let safeBatchSize = min(batchSize, memory.count)
        let memoryCount = memory.count
        
        var result = [Experience]()
        result.reserveCapacity(safeBatchSize)
        
        for _ in 0..<safeBatchSize {
            let idx = Int.random(in: 0..<memoryCount)
            result.append(memory[idx])
        }
        return result
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
    
    private var compiledStep: (([MLXArray]) -> [MLXArray])?

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
        self.memory = ReplayMemory(capacity: bufferCapacity)

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
        
        let (kRoll, kNext) = MLX.split(key: key)
        
        let roll = MLX.uniform(0..<1, key: kRoll).item() as Float
        if roll < epsilon {
            // explore: split again to get a fresh key for randInt
            let (kAction, kRemaining) = MLX.split(key: kNext)
            key = kRemaining
            let randomAction = MLX.randInt(
                low: 0,
                high: actionSize,
                [1, 1],
                key: kAction
            )
            return randomAction
        } else {
            // exploit
            key = kNext
            let shp = state.shape
            let stateRow: MLXArray
            if shp.count == 1 && shp[0] == stateSize {
                stateRow = state.reshaped([1, stateSize])
            } else if shp.count == 2 && shp[0] == 1 && shp[1] == stateSize {
                stateRow = state
            } else {
                stateRow = state.reshaped([1, stateSize])
            }
            let qValues = policyNetwork(stateRow)
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

    private func getCompiledStep() -> ([MLXArray]) -> [MLXArray] {
        if let existing = compiledStep {
            return existing
        }
        
        let gammaArray = MLXArray(gamma)
        let maxNorm = gradClipNorm
        
        // - policyNetwork: updated by optimizer
        // - targetNetwork: read for target Q-values
        // - optim: optimizer state updated during training
        let step = compile(
            inputs: [policyNetwork, targetNetwork, optim],
            outputs: [policyNetwork, optim]
        ) { [self] (arrays: [MLXArray]) -> [MLXArray] in
            let batchObs = arrays[0]
            let batchNextObs = arrays[1]
            let batchActions = arrays[2]
            let batchRewards = arrays[3]
            let batchTerminated = arrays[4]
            
            let nextQValues = targetNetwork(batchNextObs)
            let maxNextQ = nextQValues.max(axis: 1).reshaped([-1, 1])
            
            // TD target: r + γ * max_a' Q_target(s', a') * (1 - done)
            let targetQValues = stopGradient(
                batchRewards + (gammaArray * maxNextQ * (1 - batchTerminated))
            )
            
            func loss(model: QNetwork, x: MLXArray, targets: MLXArray) -> MLXArray {
                let predictedQ = model(x)
                let selectedQ = takeAlong(predictedQ, batchActions, axis: 1)
                return smoothL1Loss(predictions: selectedQ, targets: targets, reduction: .mean)
            }
            
            let lg = valueAndGrad(model: policyNetwork, loss)
            let (lossValue, grads) = lg(policyNetwork, batchObs, targetQValues)
            let (clippedGrads, gradNormValue) = clipGradNorm(gradients: grads, maxNorm: maxNorm)
            optim.update(model: policyNetwork, gradients: clippedGrads)
            
            let currentQValues = policyNetwork(batchObs)
            let meanQValue = currentQValues.max(axis: 1).mean()
            let selectedCurrentQ = takeAlong(currentQValues, batchActions, axis: 1)
            let tdError = abs(selectedCurrentQ - targetQValues).mean()
            
            return [lossValue, gradNormValue, meanQValue, tdError]
        }
        
        compiledStep = step
        return step
    }
    
    public func update() -> (
            loss: Float, meanQ: Float, gradNorm: Float, tdError: Float
        )? {
            guard memory.memory.count >= batchSize else { return nil }

            steps += 1

            let experiences = memory.sample(batchSize: batchSize)
            let batchCount = experiences.count

            let batchObs = MLX.stacked(experiences.map { $0.observation }).reshaped(
                [batchCount, stateSize])
            let batchNextObs = MLX.stacked(experiences.map { $0.nextObservation })
                .reshaped([batchCount, stateSize])
            
            let batchActionsIdx = MLX.stacked(experiences.map { $0.action })
                .reshaped([batchCount, 1])
                .asType(.int32)
            let batchRewards = MLX.stacked(experiences.map { $0.reward })
                .reshaped([batchCount, 1])
                .asType(.float32)
            let batchTerminated = MLX.stacked(experiences.map { $0.terminated })
                .reshaped([batchCount, 1])
                .asType(.float32)

            let step = getCompiledStep()
            let results = step([batchObs, batchNextObs, batchActionsIdx, batchRewards, batchTerminated])
            let lossValue = results[0]
            let gradNormValue = results[1]
            let meanQArray = results[2]
            let tdErrorArray = results[3]
            
            eval(policyNetwork, optim, lossValue, gradNormValue, meanQArray, tdErrorArray)

            updateTargetNetwork()

            return (
                lossValue.item(Float.self),
                meanQArray.item(Float.self),
                gradNormValue.item(Float.self),
                tdErrorArray.item(Float.self)
            )
        }
        
        private func updateTargetNetwork() {
            switch targetUpdateStrategy {
            case .soft(let tau):
                softUpdate(target: targetNetwork, source: policyNetwork, tau: tau)
            case .hard(let frequency):
                if steps % frequency == 0 {
                    hardUpdate(target: targetNetwork, source: policyNetwork)
                }
            }
        }
        
        private func hardUpdate(target: Module, source: Module) {
            target.update(parameters: source.parameters())
            eval(target)
        }

        /// θ′ ← τ θ + (1 −τ )θ′
        private func softUpdate(target: Module, source: Module, tau: Float) {
            let sourceParams = source.parameters()
            let targetParams = target.parameters()
            
            let flatTarget = targetParams.flattened()
            let flatSource = sourceParams.flattened()
            let sourceDict = Dictionary(uniqueKeysWithValues: flatSource)
            
            var updatedParams = [(String, MLXArray)]()
            updatedParams.reserveCapacity(flatTarget.count)
            
            let tauArray = MLXArray(tau)
            let oneMinusTau = MLXArray(1.0 - tau)
            
            for (key, targetParam) in flatTarget {
                if let sourceParam = sourceDict[key] {
                    let updated = oneMinusTau * targetParam + tauArray * sourceParam
                    updatedParams.append((key, updated))
                }
            }
            
            if let newParams = try? NestedDictionary<String, MLXArray>.unflattened(updatedParams) {
                target.update(parameters: newParams)
            }
            
            eval(target)
        }

        private func updateEpsilonSchedule() {
            guard epsilonStart != epsilonEnd else {
                epsilon = epsilonEnd
                return
            }

            let decayConstant = max(1, epsilonDecaySteps)
            let exponent = -Float(explorationSteps) / Float(decayConstant)
            let decayFactor = exp(exponent)
            let newValue = epsilonEnd
                + (epsilonStart - epsilonEnd) * decayFactor

            if epsilonStart > epsilonEnd {
                epsilon = max(epsilonEnd, newValue)
            } else {
                epsilon = min(epsilonEnd, newValue)
            }
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
