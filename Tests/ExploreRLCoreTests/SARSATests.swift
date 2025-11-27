import Testing
import MLX
@testable import ExploreRLCore

@Suite("SARSA agent")
struct SARSATests {
    
    @Test
    func testInitialization() async throws {
        let agent = SARSAAgent(
            learningRate: 0.1,
            gamma: 0.99,
            stateSize: 16,
            actionSize: 4,
            epsilon: 0.2
        )
        
        #expect(agent.learningRate == 0.1)
        #expect(agent.gamma == 0.99)
        #expect(agent.stateSize == 16)
        #expect(agent.actionSize == 4)
        #expect(agent.epsilon == 0.2)
        #expect(agent.qTable.shape == [16, 4])
    }
    
    @Test
    func testQTableInitializedToZeros() async throws {
        let agent = SARSAAgent(
            learningRate: 0.1,
            gamma: 0.99,
            stateSize: 4,
            actionSize: 2,
            epsilon: 0.1
        )
        
        let allZeros = (agent.qTable .== MLXArray(0.0)).all().item(Bool.self)
        #expect(allZeros == true)
    }
    
    @Test
    func testUpdateUsesNextAction() async throws {
        let agent = SARSAAgent(
            learningRate: 1.0, // lr=1 for exact calculation
            gamma: 0.9,
            stateSize: 4,
            actionSize: 2,
            epsilon: 0.1
        )
        
        // Set up Q-values for next state
        let customTable = MLXArray.zeros([4, 2], type: Float.self)
        customTable[1, 0] = MLXArray(10.0) // Q(s=1, a=0) = 10
        customTable[1, 1] = MLXArray(5.0)  // Q(s=1, a=1) = 5
        agent.loadQTable(customTable)
        
        // SARSA uses the actual next action, not max
        // Update with nextAction=1 (Q=5), not the max (Q=10)
        // TD target = reward + gamma * Q(s', a') = 1 + 0.9 * 5 = 5.5
        let (newQ, _) = agent.update(
            state: 0,
            action: 0,
            reward: 1.0,
            nextState: 1,
            nextAction: 1, // SARSA uses this!
            terminated: false
        )
        
        // With lr=1.0, Q should become exactly the target
        #expect(abs(newQ - 5.5) < 0.01)
    }
    
    @Test
    func testSARSAVsQLearningDifference() async throws {
        // SARSA and Q-Learning should give different results when nextAction != argmax
        let sarsaAgent = SARSAAgent(
            learningRate: 1.0,
            gamma: 0.9,
            stateSize: 4,
            actionSize: 2,
            epsilon: 0.0
        )
        
        let qAgent = QLearningAgent(
            learningRate: 1.0,
            gamma: 0.9,
            stateSize: 4,
            actionSize: 2,
            epsilon: 0.0
        )
        
        // Same Q-table for both
        let table = MLXArray.zeros([4, 2], type: Float.self)
        table[1, 0] = MLXArray(10.0) // max action
        table[1, 1] = MLXArray(2.0)  // suboptimal action
        
        sarsaAgent.loadQTable(table)
        qAgent.loadQTable(table + MLXArray(0.0)) // copy via identity operation
        
        // Update with nextAction=1 (the suboptimal action)
        let (sarsaQ, _) = sarsaAgent.update(
            state: 0, action: 0, reward: 0.0,
            nextState: 1, nextAction: 1, terminated: false
        )
        
        let (qLearningQ, _) = qAgent.update(
            state: 0, action: 0, reward: 0.0,
            nextState: 1, nextAction: 1, terminated: false
        )
        
        // SARSA: 0 + 0.9 * 2.0 = 1.8 (uses nextAction=1)
        // Q-Learning: 0 + 0.9 * 10.0 = 9.0 (uses max)
        #expect(abs(sarsaQ - 1.8) < 0.01)
        #expect(abs(qLearningQ - 9.0) < 0.01)
        #expect(sarsaQ != qLearningQ)
    }
    
    @Test
    func testUpdateWithTerminatedState() async throws {
        let agent = SARSAAgent(
            learningRate: 1.0,
            gamma: 0.9,
            stateSize: 4,
            actionSize: 2,
            epsilon: 0.1
        )
        
        // When terminated, target = reward only (no bootstrapping)
        let (newQ, _) = agent.update(
            state: 0,
            action: 0,
            reward: 7.0,
            nextState: 1,
            nextAction: 0,
            terminated: true
        )
        
        #expect(abs(newQ - 7.0) < 0.001)
    }
    
    @Test
    func testResetTable() async throws {
        let agent = SARSAAgent(
            learningRate: 0.5,
            gamma: 0.9,
            stateSize: 4,
            actionSize: 2,
            epsilon: 0.1
        )
        
        // Update to create non-zero values
        _ = agent.update(state: 0, action: 0, reward: 1.0, nextState: 1, nextAction: 0, terminated: false)
        
        let hasNonZero = (agent.qTable .!= MLXArray(0.0)).any().item(Bool.self)
        #expect(hasNonZero == true)
        
        agent.resetTable()
        
        let allZeros = (agent.qTable .== MLXArray(0.0)).all().item(Bool.self)
        #expect(allZeros == true)
    }
    
    @Test
    func testLoadQTable() async throws {
        let agent = SARSAAgent(
            learningRate: 0.1,
            gamma: 0.99,
            stateSize: 4,
            actionSize: 2,
            epsilon: 0.1
        )
        
        let customTable = MLXArray.ones([4, 2], type: Float.self) * 3.0
        agent.loadQTable(customTable)
        
        let value = agent.qTable[0, 0].item(Float.self)
        #expect(abs(value - 3.0) < 0.001)
    }
    
    @Test
    func testChooseActionExploitsWhenEpsilonZero() async throws {
        let agent = SARSAAgent(
            learningRate: 0.1,
            gamma: 0.99,
            stateSize: 4,
            actionSize: 3,
            epsilon: 0.0
        )
        
        let customTable = MLXArray.zeros([4, 3], type: Float.self)
        customTable[0, 1] = MLXArray(10.0) // action 1 is best
        agent.loadQTable(customTable)
        
        let actionSpace = Discrete(n: 3)
        var key = MLX.key(42)
        
        for _ in 0..<10 {
            let action = agent.chooseAction(actionSpace: actionSpace, state: 0, key: &key)
            #expect(action == 1)
        }
    }
    
    @Test
    func testChooseActionExploresWhenEpsilonOne() async throws {
        let agent = SARSAAgent(
            learningRate: 0.1,
            gamma: 0.99,
            stateSize: 4,
            actionSize: 4,
            epsilon: 1.0
        )
        
        let customTable = MLXArray.zeros([4, 4], type: Float.self)
        customTable[0, 0] = MLXArray(100.0)
        agent.loadQTable(customTable)
        
        let actionSpace = Discrete(n: 4)
        var key = MLX.key(999)
        
        var seenActions = Set<Int>()
        for _ in 0..<100 {
            let action = agent.chooseAction(actionSpace: actionSpace, state: 0, key: &key)
            seenActions.insert(action)
        }
        
        #expect(seenActions.count > 1)
    }
    
    @Test
    func testTDErrorCalculation() async throws {
        let agent = SARSAAgent(
            learningRate: 0.1,
            gamma: 0.9,
            stateSize: 4,
            actionSize: 2,
            epsilon: 0.1
        )
        
        let customTable = MLXArray.zeros([4, 2], type: Float.self)
        customTable[0, 0] = MLXArray(5.0)  // Q(s=0, a=0) = 5
        customTable[1, 1] = MLXArray(8.0)  // Q(s=1, a=1) = 8
        agent.loadQTable(customTable)
        
        // TD error for SARSA = reward + gamma * Q(s', a') - Q(s, a)
        // TD error = 1.0 + 0.9 * 8.0 - 5.0 = 1.0 + 7.2 - 5.0 = 3.2
        let (_, tdError) = agent.update(
            state: 0,
            action: 0,
            reward: 1.0,
            nextState: 1,
            nextAction: 1, // SARSA uses actual next action
            terminated: false
        )
        
        #expect(abs(tdError - 3.2) < 0.01)
    }
    
    @Test
    func testConformsToDiscreteRLAgent() async throws {
        let agent = SARSAAgent(
            learningRate: 0.1,
            gamma: 0.99,
            stateSize: 16,
            actionSize: 4,
            epsilon: 0.15
        )
        
        let discreteAgent: any DiscreteRLAgent = agent
        #expect(discreteAgent.epsilon == 0.15)
        #expect(discreteAgent.qTable.shape == [16, 4])
    }
}

