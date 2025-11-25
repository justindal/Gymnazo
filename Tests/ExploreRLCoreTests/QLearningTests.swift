import Testing
import MLX
@testable import ExploreRLCore

@Suite("QLearning agent")
struct QLearningTests {
    
    @Test
    func testInitialization() async throws {
        let agent = QLearningAgent(
            learningRate: 0.1,
            gamma: 0.99,
            stateSize: 16,
            actionSize: 4,
            epsilon: 0.1
        )
        
        #expect(agent.learningRate == 0.1)
        #expect(agent.gamma == 0.99)
        #expect(agent.stateSize == 16)
        #expect(agent.actionSize == 4)
        #expect(agent.epsilon == 0.1)
        #expect(agent.qTable.shape == [16, 4])
    }
    
    @Test
    func testQTableInitializedToZeros() async throws {
        let agent = QLearningAgent(
            learningRate: 0.1,
            gamma: 0.99,
            stateSize: 4,
            actionSize: 2,
            epsilon: 0.1
        )
        
        // All Q-values should be zero initially
        let allZeros = (agent.qTable .== MLXArray(0.0)).all().item(Bool.self)
        #expect(allZeros == true)
    }
    
    @Test
    func testUpdateIncreasesQValue() async throws {
        let agent = QLearningAgent(
            learningRate: 0.5,
            gamma: 0.9,
            stateSize: 4,
            actionSize: 2,
            epsilon: 0.1
        )
        
        let initialQ = agent.qTable[0, 0].item(Float.self)
        #expect(initialQ == 0.0)
        
        // Perform an update with positive reward
        let (newQ, tdError) = agent.update(
            state: 0,
            action: 0,
            reward: 1.0,
            nextState: 1,
            nextAction: 0, // ignored by Q-learning (uses max)
            terminated: false
        )
        
        #expect(newQ > initialQ)
        #expect(tdError > 0)
    }
    
    @Test
    func testUpdateWithTerminatedState() async throws {
        let agent = QLearningAgent(
            learningRate: 1.0, // lr=1 for easier verification
            gamma: 0.9,
            stateSize: 4,
            actionSize: 2,
            epsilon: 0.1
        )
        
        // When terminated, target should be just the reward (no future value)
        let (newQ, _) = agent.update(
            state: 0,
            action: 0,
            reward: 10.0,
            nextState: 1,
            nextAction: 0,
            terminated: true
        )
        
        // With lr=1.0 and terminated=true, Q(s,a) should become exactly the reward
        #expect(abs(newQ - 10.0) < 0.001)
    }
    
    @Test
    func testResetTable() async throws {
        let agent = QLearningAgent(
            learningRate: 0.5,
            gamma: 0.9,
            stateSize: 4,
            actionSize: 2,
            epsilon: 0.1
        )
        
        // Update to create non-zero values
        _ = agent.update(state: 0, action: 0, reward: 1.0, nextState: 1, nextAction: 0, terminated: false)
        
        // Verify Q-table has non-zero values
        let hasNonZero = (agent.qTable .!= MLXArray(0.0)).any().item(Bool.self)
        #expect(hasNonZero == true)
        
        // Reset
        agent.resetTable()
        
        // Verify all zeros again
        let allZeros = (agent.qTable .== MLXArray(0.0)).all().item(Bool.self)
        #expect(allZeros == true)
    }
    
    @Test
    func testLoadQTable() async throws {
        let agent = QLearningAgent(
            learningRate: 0.1,
            gamma: 0.99,
            stateSize: 4,
            actionSize: 2,
            epsilon: 0.1
        )
        
        // Create a custom Q-table
        let customTable = MLXArray.ones([4, 2], type: Float.self) * 5.0
        agent.loadQTable(customTable)
        
        // Verify it was loaded
        let value = agent.qTable[0, 0].item(Float.self)
        #expect(abs(value - 5.0) < 0.001)
    }
    
    @Test
    func testChooseActionExploitsWhenEpsilonZero() async throws {
        let agent = QLearningAgent(
            learningRate: 0.1,
            gamma: 0.99,
            stateSize: 4,
            actionSize: 3,
            epsilon: 0.0 // no exploration
        )
        
        // Set Q-values so action 2 is best for state 0
        var customTable = MLXArray.zeros([4, 3], type: Float.self)
        customTable[0, 2] = MLXArray(10.0)
        agent.loadQTable(customTable)
        
        let actionSpace = Discrete(n: 3)
        var key = MLX.key(42)
        
        // With epsilon=0, should always choose action 2
        for _ in 0..<10 {
            let action = agent.chooseAction(actionSpace: actionSpace, state: 0, key: &key)
            #expect(action == 2)
        }
    }
    
    @Test
    func testChooseActionExploresWhenEpsilonOne() async throws {
        let agent = QLearningAgent(
            learningRate: 0.1,
            gamma: 0.99,
            stateSize: 4,
            actionSize: 4,
            epsilon: 1.0 // always explore
        )
        
        // Set Q-values so action 0 would be best
        var customTable = MLXArray.zeros([4, 4], type: Float.self)
        customTable[0, 0] = MLXArray(100.0)
        agent.loadQTable(customTable)
        
        let actionSpace = Discrete(n: 4)
        var key = MLX.key(123)
        
        // With epsilon=1.0, should explore and hit different actions
        var seenActions = Set<Int>()
        for _ in 0..<100 {
            let action = agent.chooseAction(actionSpace: actionSpace, state: 0, key: &key)
            seenActions.insert(action)
        }
        
        // Should have seen multiple different actions due to exploration
        #expect(seenActions.count > 1)
    }
    
    @Test
    func testTDErrorCalculation() async throws {
        let agent = QLearningAgent(
            learningRate: 0.1,
            gamma: 0.9,
            stateSize: 4,
            actionSize: 2,
            epsilon: 0.1
        )
        
        // Set up known Q-values
        var customTable = MLXArray.zeros([4, 2], type: Float.self)
        customTable[0, 0] = MLXArray(5.0)  // Q(s=0, a=0) = 5
        customTable[1, 0] = MLXArray(10.0) // Q(s=1, a=0) = 10 (max for next state)
        customTable[1, 1] = MLXArray(8.0)  // Q(s=1, a=1) = 8
        agent.loadQTable(customTable)
        
        // TD error = reward + gamma * max(Q(s')) - Q(s, a)
        // TD error = 1.0 + 0.9 * 10.0 - 5.0 = 1.0 + 9.0 - 5.0 = 5.0
        let (_, tdError) = agent.update(
            state: 0,
            action: 0,
            reward: 1.0,
            nextState: 1,
            nextAction: 1, // ignored - Q-learning uses max
            terminated: false
        )
        
        #expect(abs(tdError - 5.0) < 0.01)
    }
    
    @Test
    func testConformsToDiscreteRLAgent() async throws {
        let agent = QLearningAgent(
            learningRate: 0.1,
            gamma: 0.99,
            stateSize: 16,
            actionSize: 4,
            epsilon: 0.1
        )
        
        // Verify it can be used as DiscreteRLAgent
        let discreteAgent: any DiscreteRLAgent = agent
        #expect(discreteAgent.epsilon == 0.1)
        #expect(discreteAgent.qTable.shape == [16, 4])
    }
}

