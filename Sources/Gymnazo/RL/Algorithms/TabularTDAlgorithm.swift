//
//  TabularTDAlgorithm.swift
//  Gymnazo
//

import MLX

public protocol TabularTDAlgorithm: AnyObject {
    var config: TabularTDConfig { get }
    var env: (any Env)? { get set }
    var nStates: Int { get }
    var nActions: Int { get }
    var actionSpace: Discrete { get }
    var qTable: MLXArray { get set }
    var explorationRate: Double { get set }
    var randomKey: MLXArray { get set }
    var numTimesteps: Int { get set }
    var totalTimesteps: Int { get set }
    
    var requiresNextAction: Bool { get }
    
    func nextQ(nextState: MLXArray, nextAction: MLXArray?) -> MLXArray
}

public struct TabularTDConfig: Sendable {
    public let learningRate: Double
    public let gamma: Double
    public let explorationFraction: Double
    public let explorationInitialEps: Double
    public let explorationFinalEps: Double

    public init(
        learningRate: Double = 0.5,
        gamma: Double = 0.95,
        explorationFraction: Double = 0.5,
        explorationInitialEps: Double = 1.0,
        explorationFinalEps: Double = 0.1
    ) {
        self.learningRate = learningRate
        self.gamma = gamma
        self.explorationFraction = explorationFraction
        self.explorationInitialEps = explorationInitialEps
        self.explorationFinalEps = explorationFinalEps
    }
}

extension TabularTDAlgorithm {
    @discardableResult
    public func learn(totalTimesteps: Int) throws -> Self {
        self.totalTimesteps = totalTimesteps
        self.numTimesteps = 0

        guard var environment = env else {
            throw GymnazoError.invalidState(
                "\(Self.self).learn requires an environment. Set env before calling learn()."
            )
        }

        var state = try environment.reset().obs
        var action = selectAction(state: state, forExploration: true)

        while numTimesteps < totalTimesteps {
            updateExplorationRate()

            if !requiresNextAction {
                action = selectAction(state: state, forExploration: true)
            }

            let stepResult = try environment.step(action)
            let reward = stepResult.reward
            let nextState = stepResult.obs
            let terminated = stepResult.terminated
            let truncated = stepResult.truncated

            let nextAction: MLXArray?
            if requiresNextAction && !terminated && !truncated {
                nextAction = selectAction(state: nextState, forExploration: true)
            } else {
                nextAction = nil
            }

            updateQValue(
                state: state,
                action: action,
                reward: reward,
                nextState: nextState,
                nextAction: nextAction,
                terminated: terminated
            )

            if terminated || truncated {
                state = try environment.reset().obs
                if requiresNextAction {
                    action = selectAction(state: state, forExploration: true)
                }
            } else {
                state = nextState
                if requiresNextAction, let next = nextAction {
                    action = next
                }
            }

            numTimesteps += 1
        }

        self.env = environment
        return self
    }

    public func updateExplorationRate() {
        let fraction = min(
            1.0,
            Double(numTimesteps) / (Double(totalTimesteps) * config.explorationFraction)
        )
        explorationRate =
            config.explorationInitialEps
            + fraction * (config.explorationFinalEps - config.explorationInitialEps)
    }

    public func selectAction(state: MLXArray, forExploration: Bool) -> MLXArray {
        if forExploration {
            let (exploreKey, nextKey) = MLX.split(key: randomKey)
            randomKey = nextKey

            if shouldExplore(key: exploreKey) {
                let (sampleKey, nextKey2) = MLX.split(key: randomKey)
                randomKey = nextKey2
                return actionSpace.sample(key: sampleKey)
            }
        }
        return selectBestAction(state: state)
    }

    public func shouldExplore(key: MLXArray) -> Bool {
        let random = MLX.uniform(0.0..<1.0, key: key)
        eval(random)
        let value: Float = random.item()
        return Double(value) < explorationRate
    }

    public func selectBestAction(state: MLXArray) -> MLXArray {
        let qValues = getQValues(for: state)
        return MLX.argMax(qValues).asType(.int32)
    }

    public func updateQValue(
        state: MLXArray,
        action: MLXArray,
        reward: Double,
        nextState: MLXArray,
        nextAction: MLXArray?,
        terminated: Bool
    ) {
        let currentQ = getQValue(state: state, action: action)

        let nextQ: MLXArray
        if terminated {
            nextQ = MLXArray(0.0)
        } else {
            nextQ = self.nextQ(nextState: nextState, nextAction: nextAction)
        }

        let rewardArray = MLXArray(reward)
        let gamma = MLXArray(config.gamma)
        let alpha = MLXArray(config.learningRate)
        
        let tdTarget = rewardArray + gamma * nextQ
        let newQ = currentQ + alpha * (tdTarget - currentQ)
        eval(newQ)

        setQValue(state: state, action: action, value: newQ)
    }

    public func getQValues(for state: MLXArray) -> MLXArray {
        let stateIdx = Int(state.asType(.int32).item(Int32.self))
        return qTable[stateIdx]
    }

    public func getQValue(state: MLXArray, action: MLXArray) -> MLXArray {
        let stateIdx = Int(state.asType(.int32).item(Int32.self))
        let actionIdx = Int(action.asType(.int32).item(Int32.self))
        return qTable[stateIdx, actionIdx]
    }

    public func setQValue(state: MLXArray, action: MLXArray, value: MLXArray) {
        let stateIdx = Int(state.asType(.int32).item(Int32.self))
        let actionIdx = Int(action.asType(.int32).item(Int32.self))
        
        var tableData = qTable.asArray(Float.self)
        let flatIdx = stateIdx * nActions + actionIdx
        tableData[flatIdx] = value.item(Float.self)
        qTable = MLXArray(tableData).reshaped([nStates, nActions])
        eval(qTable)
    }

    public var currentExplorationRate: Double {
        explorationRate
    }

    public func predict(observation: MLXArray, deterministic: Bool = true) -> MLXArray {
        if !deterministic {
            let (exploreKey, nextKey) = MLX.split(key: randomKey)
            randomKey = nextKey

            if shouldExplore(key: exploreKey) {
                let (sampleKey, nextKey2) = MLX.split(key: randomKey)
                randomKey = nextKey2
                return actionSpace.sample(key: sampleKey)
            }
        }
        return selectBestAction(state: observation)
    }

    public func qValues(for state: MLXArray) -> MLXArray {
        getQValues(for: state)
    }

    public var table: MLXArray {
        qTable
    }
}
