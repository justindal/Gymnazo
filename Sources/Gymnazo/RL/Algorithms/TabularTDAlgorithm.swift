//
//  TabularTDAlgorithm.swift
//  Gymnazo
//

import MLX

public protocol TabularTDAlgorithm: AnyObject {
    associatedtype Environment: Env where Environment.Observation == Int, Environment.Action == Int
    
    var config: TabularTDConfig { get }
    var env: Environment? { get set }
    var nActions: Int { get }
    var actionSpace: Discrete { get }
    var qTable: [Int: [Double]] { get set }
    var explorationRate: Double { get set }
    var randomKey: MLXArray { get set }
    var numTimesteps: Int { get set }
    var totalTimesteps: Int { get set }
    
    var requiresNextAction: Bool { get }
    
    func computeNextQ(nextState: Int, nextAction: Int?) -> Double
}

public struct TabularTDConfig: Sendable {
    public let learningRate: Double
    public let gamma: Double
    public let explorationFraction: Double
    public let explorationInitialEps: Double
    public let explorationFinalEps: Double

    public init(
        learningRate: Double = 0.1,
        gamma: Double = 0.99,
        explorationFraction: Double = 0.1,
        explorationInitialEps: Double = 1.0,
        explorationFinalEps: Double = 0.05
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

            let nextAction: Int?
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

    public func selectAction(state: Int, forExploration: Bool) -> Int {
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
        MLX.eval(random)
        let value: Float = random.item()
        return Double(value) < explorationRate
    }

    public func selectBestAction(state: Int) -> Int {
        let qValues = getQValues(for: state)
        return qValues.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
    }

    public func updateQValue(
        state: Int,
        action: Int,
        reward: Double,
        nextState: Int,
        nextAction: Int?,
        terminated: Bool
    ) {
        let currentQ = getQValue(state: state, action: action)

        let nextQ: Double
        if terminated {
            nextQ = 0.0
        } else {
            nextQ = computeNextQ(nextState: nextState, nextAction: nextAction)
        }

        let tdTarget = reward + config.gamma * nextQ
        let newQ = currentQ + config.learningRate * (tdTarget - currentQ)

        setQValue(state: state, action: action, value: newQ)
    }

    public func getQValues(for state: Int) -> [Double] {
        if let values = qTable[state] {
            return values
        }
        let zeros = [Double](repeating: 0.0, count: nActions)
        qTable[state] = zeros
        return zeros
    }

    public func getQValue(state: Int, action: Int) -> Double {
        getQValues(for: state)[action]
    }

    public func setQValue(state: Int, action: Int, value: Double) {
        if qTable[state] == nil {
            qTable[state] = [Double](repeating: 0.0, count: nActions)
        }
        qTable[state]![action] = value
    }

    public var currentExplorationRate: Double {
        explorationRate
    }

    public func predict(state: Int, deterministic: Bool = true) -> Int {
        if !deterministic {
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

    public func qValues(for state: Int) -> [Double] {
        getQValues(for: state)
    }

    public var table: [Int: [Double]] {
        qTable
    }
}
