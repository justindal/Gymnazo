import MLX

public actor TabularAgent {
    public enum UpdateRule: String, Sendable, Codable {
        case qLearning
        case sarsa
    }

    public let updateRule: UpdateRule
    public let config: TabularConfig
    public let numStates: Int
    public let numActions: Int

    nonisolated(unsafe) private var env: (any Env)?
    private let actionSpace: Discrete
    private let stateStrides: [Int]
    private var qTable: MLXArray
    private var key: MLXArray
    private var explorationRate: Double
    private var timesteps: Int = 0
    private var totalTimesteps: Int = 0
    private var shouldContinue: Bool = true

    public init(
        updateRule: UpdateRule,
        config: TabularConfig,
        numStates: Int,
        numActions: Int,
        seed: UInt64? = nil,
        stateStrides: [Int] = []
    ) {
        self.env = nil
        self.updateRule = updateRule
        self.config = config
        self.numStates = numStates
        self.numActions = numActions
        self.actionSpace = Discrete(n: numActions)
        self.stateStrides = stateStrides
        self.qTable = MLX.zeros([numStates, numActions])
        self.explorationRate = config.epsilon
        self.key = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
    }

    public init(
        updateRule: UpdateRule,
        config: TabularConfig,
        numStates: Int,
        numActions: Int,
        seed: UInt64? = nil,
        stateStrides: [Int] = [],
        qTable: MLXArray,
        timesteps: Int,
        explorationRate: Double
    ) {
        self.env = nil
        self.updateRule = updateRule
        self.config = config
        self.numStates = numStates
        self.numActions = numActions
        self.actionSpace = Discrete(n: numActions)
        self.stateStrides = stateStrides
        self.qTable = qTable
        self.timesteps = timesteps
        self.explorationRate = explorationRate
        self.key = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
    }

    public init(
        env: any Env,
        updateRule: UpdateRule,
        config: TabularConfig = TabularConfig(),
        seed: UInt64? = nil
    ) {
        guard let act = env.actionSpace as? Discrete else {
            preconditionFailure("TabularAgent requires a Discrete action space.")
        }

        let info = Self.stateSpaceInfo(from: env.observationSpace)

        self.env = env
        self.updateRule = updateRule
        self.config = config
        self.numStates = info.numStates
        self.numActions = act.n
        self.actionSpace = Discrete(n: act.n)
        self.stateStrides = info.strides
        self.qTable = MLX.zeros([info.numStates, act.n])
        self.explorationRate = config.epsilon
        self.key = MLX.key(seed ?? UInt64.random(in: 0..<UInt64.max))
    }

    public static func stateSpaceInfo(
        from observationSpace: any Space
    ) -> (numStates: Int, strides: [Int]) {
        if let discrete = observationSpace as? Discrete {
            return (discrete.n, [])
        }
        if let tuple = observationSpace as? Tuple {
            let dims = tuple.spaces.compactMap { ($0 as? Discrete)?.n }
            precondition(
                dims.count == tuple.spaces.count,
                "All Tuple components must be Discrete for tabular methods.")
            var strides = Array(repeating: 1, count: dims.count)
            for i in stride(from: dims.count - 2, through: 0, by: -1) {
                strides[i] = strides[i + 1] * dims[i + 1]
            }
            return (dims.reduce(1, *), strides)
        }
        preconditionFailure(
            "TabularAgent requires Discrete or Tuple(Discrete...) observation space.")
    }

    public func learn(
        totalTimesteps: Int,
        callbacks: LearnCallbacks?,
        resetProgress: Bool = true
    ) async throws {
        self.totalTimesteps = totalTimesteps
        if resetProgress { self.timesteps = 0 }
        shouldContinue = true

        guard var environment = env else {
            throw GymnazoError.invalidState(
                "TabularAgent.learn() requires an environment."
            )
        }

        var state = flatIndex(for: try environment.reset().obs)
        var action = selectAction(state: state)
        var episodeReward: Double = 0
        var episodeLength: Int = 0

        while timesteps < totalTimesteps {
            guard shouldContinue else { break }

            if updateRule == .qLearning {
                action = selectAction(state: state)
            }

            let step = try environment.step(action)
            let reward = step.reward
            let nextState = flatIndex(for: step.obs)
            let terminated = step.terminated
            let truncated = step.truncated

            episodeReward += reward
            episodeLength += 1

            let currentQ = qTable[state, action]
            let futureQ: MLXArray
            var nextAction: MLXArray?

            switch (terminated, updateRule) {
            case (true, _):
                futureQ = MLXArray(0.0)
            case (false, .qLearning):
                futureQ = qTable[nextState].max(stream: .cpu)
            case (false, .sarsa):
                let nextA = selectAction(state: nextState)
                futureQ = qTable[nextState, nextA]
                nextAction = nextA
            }

            let newQ = tdUpdate(currentQ, reward: reward, futureQ)
            qTable[state, action] = newQ
            eval(qTable)

            if terminated || truncated {
                explorationRate = max(config.minEpsilon, explorationRate * config.epsilonDecay)

                await callbacks?.onEpisodeEnd?(episodeReward, episodeLength)
                episodeReward = 0
                episodeLength = 0

                state = flatIndex(for: try environment.reset().obs)
                action = selectAction(state: state)
            } else {
                state = nextState
                if let nextA = nextAction {
                    action = nextA
                }
            }

            timesteps += 1

            if let onStep = callbacks?.onStep {
                let cont = await onStep(timesteps, totalTimesteps, explorationRate)
                if !cont { break }
            }

            if let onSnapshot = callbacks?.onSnapshot {
                if let output = try? environment.render() {
                    if case .other(let snapshot) = output {
                        await onSnapshot(snapshot)
                    }
                }
            }
        }

        self.env = environment
    }

    public func evaluate(
        episodes: Int,
        deterministic: Bool = true,
        callbacks: EvaluateCallbacks? = nil
    ) async throws {
        guard var environment = env else {
            throw GymnazoError.invalidState("TabularAgent.evaluate requires an environment.")
        }

        for _ in 0..<episodes {
            var obs = try environment.reset().obs
            var done = false
            var episodeReward: Double = 0
            var episodeLength: Int = 0

            while !done {
                let stateIdx: Int32 = flatIndex(for: obs).item()
                let actionIndex = predict(state: stateIdx, deterministic: deterministic)
                let action = MLXArray(actionIndex)

                let step = try environment.step(action)
                episodeReward += step.reward
                episodeLength += 1
                done = step.terminated || step.truncated
                obs = step.obs

                if let onSnapshot = callbacks?.onSnapshot {
                    if let output = try? environment.render() {
                        if case .other(let snapshot) = output {
                            await onSnapshot(snapshot)
                        }
                    }
                }

                if let onStep = callbacks?.onStep {
                    if !(await onStep()) {
                        self.env = environment
                        return
                    }
                }
            }

            await callbacks?.onEpisodeEnd?(episodeReward, episodeLength)
        }

        self.env = environment
    }

    public func predict(state: Int32, deterministic: Bool = true) -> Int32 {
        if !deterministic {
            var k = nextKey(for: &key, stream: .cpu)
            let random = MLX.uniform(0.0..<1.0, key: k, stream: .cpu)
            eval(random)
            if Double(random.item(Float.self)) < explorationRate {
                k = nextKey(for: &key, stream: .cpu)
                let sample = actionSpace.sample(key: k)
                eval(sample)
                return sample.item()
            }
        }
        let action = MLX.argMax(qTable[Int(state)], stream: .cpu).asType(.int32, stream: .cpu)
        eval(action)
        return action.item()
    }

    public func stop() {
        shouldContinue = false
    }

    public var numTimesteps: Int { timesteps }
    public var epsilon: Double { explorationRate }
    var table: MLXArray { qTable }
    var strides: [Int] { stateStrides }

    public func tableValues() -> [Float] {
        eval(qTable)
        return qTable.asArray(Float.self)
    }

    nonisolated public func setEnv(_ env: any Env) {
        self.env = env
    }

    nonisolated public func takeEnv() -> (any Env)? {
        let e = env
        env = nil
        return e
    }

    public func restore(timesteps: Int, explorationRate: Double, qTable: MLXArray) {
        self.timesteps = timesteps
        self.explorationRate = explorationRate
        self.qTable = qTable
    }

    private func tdUpdate(_ currentQ: MLXArray, reward: Double, _ futureQ: MLXArray) -> MLXArray {
        Device.withDefaultDevice(.cpu) {
            currentQ + config.learningRate * (reward + config.gamma * futureQ - currentQ)
        }
    }

    private func selectAction(state: MLXArray) -> MLXArray {
        var k = nextKey(for: &key, stream: .cpu)
        let random = MLX.uniform(0.0..<1.0, key: k, stream: .cpu)
        eval(random)

        if Double(random.item(Float.self)) < explorationRate {
            k = nextKey(for: &key, stream: .cpu)
            return actionSpace.sample(key: k)
        }
        return MLX.argMax(qTable[state], stream: .cpu).asType(.int32, stream: .cpu)
    }

    private func flatIndex(for obs: MLXArray) -> MLXArray {
        guard !stateStrides.isEmpty else { return obs }
        eval(obs)
        let values = obs.asArray(Int32.self)
        var idx: Int32 = 0
        for (v, s) in zip(values, stateStrides) {
            idx += v * Int32(s)
        }
        return MLXArray(idx)
    }
}
