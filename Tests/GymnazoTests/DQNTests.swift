import Testing
import MLX
import MLXNN
import MLXOptimizers
@testable import Gymnazo

@Suite("DQN components")
struct DQNTests {
    
    @Suite("QNetwork")
    struct QNetworkTests {
        
        @Test
        func testInitialization() async throws {
            let network = QNetwork(numObservations: 4, numActions: 2, hiddenSize: 64)
            
            let input = MLXArray.zeros([1, 4], type: Float.self)
            let output = network(input)
            
            #expect(output.shape == [1, 2])
        }
        
        @Test
        func testOutputShape() async throws {
            let network = QNetwork(numObservations: 8, numActions: 4, hiddenSize: 128)
            
            let input = MLXArray.zeros([5, 8], type: Float.self)
            let output = network(input)
            
            #expect(output.shape == [5, 4])
        }
        
        @Test
        func testDifferentHiddenSizes() async throws {
            let smallNet = QNetwork(numObservations: 4, numActions: 2, hiddenSize: 32)
            let largeNet = QNetwork(numObservations: 4, numActions: 2, hiddenSize: 256)
            
            let input = MLXArray.zeros([1, 4], type: Float.self)
            
            let smallOut = smallNet(input)
            let largeOut = largeNet(input)
            
            #expect(smallOut.shape == [1, 2])
            #expect(largeOut.shape == [1, 2])
        }
        
        @Test
        func testForwardPassProducesFiniteValues() async throws {
            let network = QNetwork(numObservations: 4, numActions: 2)
            
            let input = MLX.uniform(low: Float(-1.0), high: Float(1.0), [10, 4])
            let output = network(input)
            eval(output)
            
            let allFinite = (output .== output).all().item(Bool.self)
            #expect(allFinite == true)
        }
    }
    
    @Suite("ReplayMemory")
    struct ReplayMemoryTests {
        
        private func makeExperience(value: Float = 0.0) -> Experience {
            Experience(
                observation: MLXArray([value, value, value, value]),
                nextObservation: MLXArray([value + 1, value + 1, value + 1, value + 1]),
                action: MLXArray(Int32(0)),
                reward: MLXArray(value),
                terminated: MLXArray(Float(0.0))
            )
        }
        
        @Test
        func testInitialization() async throws {
            let memory = ReplayMemory(capacity: 100)
            #expect(memory.capacity == 100)
            #expect(memory.memory.isEmpty == true)
        }
        
        @Test
        func testPushAndCount() async throws {
            let memory = ReplayMemory(capacity: 10)
            
            memory.push(makeExperience(value: 1.0))
            #expect(memory.memory.count == 1)
            
            memory.push(makeExperience(value: 2.0))
            #expect(memory.memory.count == 2)
        }
        
        @Test
        func testCapacityLimit() async throws {
            let memory = ReplayMemory(capacity: 3)
            
            for i in 0..<5 {
                memory.push(makeExperience(value: Float(i)))
            }
            
            #expect(memory.memory.count == 3)
        }
        
        @Test
        func testFIFORemoval() async throws {
            let memory = ReplayMemory(capacity: 3)
            
            for i in 0..<5 {
                memory.push(makeExperience(value: Float(i)))
            }
            
            let rewards = memory.memory.map { $0.reward.item(Float.self) }
            #expect(rewards.contains(2.0))
            #expect(rewards.contains(3.0))
            #expect(rewards.contains(4.0))
            #expect(rewards.contains(0.0) == false)
            #expect(rewards.contains(1.0) == false)
        }
        
        @Test
        func testSampleBatchSize() async throws {
            let memory = ReplayMemory(capacity: 100)
            
            for i in 0..<50 {
                memory.push(makeExperience(value: Float(i)))
            }
            
            let batch = memory.sample(batchSize: 10)
            #expect(batch.count == 10)
        }
        
        @Test
        func testSampleClampsToAvailable() async throws {
            let memory = ReplayMemory(capacity: 100)
            
            // Only add 5 experiences
            for i in 0..<5 {
                memory.push(makeExperience(value: Float(i)))
            }
            
            // Request 20, should only get 5
            let batch = memory.sample(batchSize: 20)
            #expect(batch.count == 5)
        }
    }
    
    @Suite("DQNAgent")
    struct DQNAgentTests {
        
        @Test
        func testInitialization() async throws {
            let agent = DQNAgent(
                batchSize: 32,
                stateSize: 4,
                actionSize: 2,
                gamma: 0.99,
                epsilonStart: 1.0,
                epsilonEnd: 0.01,
                epsilonDecaySteps: 1000,
                targetUpdateStrategy: .soft(tau: 0.005),
                learningRate: 0.001,
                optim: AdamW(learningRate: 0.001),
                bufferCapacity: 10000
            )
            
            #expect(agent.stateSize == 4)
            #expect(agent.actionSize == 2)
            #expect(agent.gamma == 0.99)
            #expect(agent.epsilon == 1.0)
            #expect(agent.batchSize == 32)
        }
        
        @Test
        func testConvenienceInitializer() async throws {
            let obsSpace = Box(
                low: MLXArray([-1.0, -1.0, -1.0, -1.0] as [Float]),
                high: MLXArray([1.0, 1.0, 1.0, 1.0] as [Float]),
                dtype: .float32
            )
            let actionSpace = Discrete(n: 2)
            
            let agent = DQNAgent(
                observationSpace: obsSpace,
                actionSpace: actionSpace,
                hiddenDimensions: 64,
                learningRate: 0.001,
                gamma: 0.99,
                epsilon: 0.9,
                epsilonEnd: 0.05,
                epsilonDecaySteps: 500,
                tau: 0.01,
                batchSize: 64,
                bufferSize: 5000,
                gradClipNorm: 10.0
            )
            
            #expect(agent.stateSize == 4)
            #expect(agent.actionSize == 2)
        }
        
        @Test
        func testStoreExperience() async throws {
            let agent = DQNAgent(
                batchSize: 32,
                stateSize: 4,
                actionSize: 2,
                gamma: 0.99,
                epsilonStart: 1.0,
                epsilonEnd: 0.01,
                epsilonDecaySteps: 1000,
                learningRate: 0.001,
                optim: AdamW(learningRate: 0.001)
            )
            
            let state = MLXArray([0.1, 0.2, 0.3, 0.4] as [Float])
            let action = MLXArray(Int32(1))
            let nextState = MLXArray([0.2, 0.3, 0.4, 0.5] as [Float])
            
            agent.store(state: state, action: action, reward: 1.0, nextState: nextState, terminated: false)
            
            #expect(agent.memory.memory.count == 1)
        }
        
        @Test
        func testChooseActionShape() async throws {
            let agent = DQNAgent(
                batchSize: 32,
                stateSize: 4,
                actionSize: 3,
                gamma: 0.99,
                epsilonStart: 0.0,
                epsilonEnd: 0.0,
                epsilonDecaySteps: 1,
                learningRate: 0.001,
                optim: AdamW(learningRate: 0.001)
            )
            
            let state = MLXArray([0.1, 0.2, 0.3, 0.4] as [Float])
            let actionSpace = Discrete(n: 3)
            var key = MLX.key(42)
            
            let action = agent.chooseAction(state: state, actionSpace: actionSpace, key: &key)
            eval(action)
            
            #expect(action.shape == [1, 1])
            
            // Action should be valid (0, 1, or 2)
            let actionVal = action.item(Int.self)
            #expect(actionVal >= 0 && actionVal < 3)
        }
        
        @Test
        func testEpsilonDecay() async throws {
            let agent = DQNAgent(
                batchSize: 32,
                stateSize: 4,
                actionSize: 2,
                gamma: 0.99,
                epsilonStart: 1.0,
                epsilonEnd: 0.1,
                epsilonDecaySteps: 100,
                learningRate: 0.001,
                optim: AdamW(learningRate: 0.001)
            )
            
            let initialEpsilon = agent.epsilon
            #expect(initialEpsilon == 1.0)
            
            let state = MLXArray([0.1, 0.2, 0.3, 0.4] as [Float])
            let actionSpace = Discrete(n: 2)
            var key = MLX.key(123)
            
            for _ in 0..<50 {
                _ = agent.chooseAction(state: state, actionSpace: actionSpace, key: &key)
            }
            
            // Epsilon should have decayed
            #expect(agent.epsilon < initialEpsilon)
            #expect(agent.epsilon >= 0.1) // shouldn't go below end
        }
        
        @Test
        func testUpdateReturnsNilWhenBufferTooSmall() async throws {
            let agent = DQNAgent(
                batchSize: 32,
                stateSize: 4,
                actionSize: 2,
                gamma: 0.99,
                epsilonStart: 1.0,
                epsilonEnd: 0.01,
                epsilonDecaySteps: 1000,
                learningRate: 0.001,
                optim: AdamW(learningRate: 0.001)
            )
            
            // Add only 10 experiences (less than batch size of 32)
            for i in 0..<10 {
                let state = MLXArray([Float(i), 0.0, 0.0, 0.0])
                let nextState = MLXArray([Float(i + 1), 0.0, 0.0, 0.0])
                agent.store(state: state, action: MLXArray(Int32(0)), reward: 1.0, nextState: nextState, terminated: false)
            }
            
            let result = agent.update()
            #expect(result == nil)
        }
        
        @Test
        func testUpdateReturnsMetricsWhenBufferSufficient() async throws {
            let agent = DQNAgent(
                batchSize: 8, // small batch for test
                stateSize: 4,
                actionSize: 2,
                gamma: 0.99,
                epsilonStart: 1.0,
                epsilonEnd: 0.01,
                epsilonDecaySteps: 1000,
                learningRate: 0.001,
                optim: AdamW(learningRate: 0.001),
                bufferCapacity: 100
            )
            
            for i in 0..<20 {
                let state = MLX.uniform(low: Float(-1.0), high: Float(1.0), [4])
                let nextState = MLX.uniform(low: Float(-1.0), high: Float(1.0), [4])
                agent.store(
                    state: state,
                    action: MLXArray(Int32(i % 2)),
                    reward: Float(i % 2),
                    nextState: nextState,
                    terminated: i == 19
                )
            }
            
            let result = agent.update()
            #expect(result != nil)
            
            if let metrics = result {
                // Check all metrics are finite
                #expect(metrics.loss.isFinite)
                #expect(metrics.meanQ.isFinite)
                #expect(metrics.gradNorm.isFinite)
                #expect(metrics.tdError.isFinite)
            }
        }
        
        @Test
        func testStepCountIncrementsOnUpdate() async throws {
            let agent = DQNAgent(
                batchSize: 4,
                stateSize: 4,
                actionSize: 2,
                gamma: 0.99,
                epsilonStart: 1.0,
                epsilonEnd: 0.01,
                epsilonDecaySteps: 1000,
                learningRate: 0.001,
                optim: AdamW(learningRate: 0.001)
            )
            
            #expect(agent.currentSteps == 0)
            
            for _ in 0..<10 {
                let state = MLX.uniform(low: Float(-1.0), high: Float(1.0), [4])
                let nextState = MLX.uniform(low: Float(-1.0), high: Float(1.0), [4])
                agent.store(state: state, action: MLXArray(Int32(0)), reward: 1.0, nextState: nextState, terminated: false)
            }
            
            _ = agent.update()
            #expect(agent.currentSteps == 1)
            
            _ = agent.update()
            #expect(agent.currentSteps == 2)
        }
        
        @Test
        func testExplorationStepTracking() async throws {
            let agent = DQNAgent(
                batchSize: 32,
                stateSize: 4,
                actionSize: 2,
                gamma: 0.99,
                epsilonStart: 1.0,
                epsilonEnd: 0.01,
                epsilonDecaySteps: 1000,
                learningRate: 0.001,
                optim: AdamW(learningRate: 0.001)
            )
            
            #expect(agent.currentExplorationSteps == 0)
            
            let state = MLXArray([0.1, 0.2, 0.3, 0.4] as [Float])
            let actionSpace = Discrete(n: 2)
            var key = MLX.key(42)
            
            _ = agent.chooseAction(state: state, actionSpace: actionSpace, key: &key)
            #expect(agent.currentExplorationSteps == 1)
            
            _ = agent.chooseAction(state: state, actionSpace: actionSpace, key: &key)
            #expect(agent.currentExplorationSteps == 2)
        }
        
        @Test
        func testSetExplorationSteps() async throws {
            let agent = DQNAgent(
                batchSize: 32,
                stateSize: 4,
                actionSize: 2,
                gamma: 0.99,
                epsilonStart: 1.0,
                epsilonEnd: 0.1,
                epsilonDecaySteps: 100,
                learningRate: 0.001,
                optim: AdamW(learningRate: 0.001)
            )
            
            // Set exploration steps to simulate resuming training
            agent.setExplorationSteps(50)
            #expect(agent.currentExplorationSteps == 50)
            
            // Epsilon should be updated based on the new step count
            #expect(agent.epsilon < 1.0)
        }
        
        @Test
        func testTargetUpdateStrategySoft() async throws {
            let agent = DQNAgent(
                batchSize: 4,
                stateSize: 4,
                actionSize: 2,
                gamma: 0.99,
                epsilonStart: 1.0,
                epsilonEnd: 0.01,
                epsilonDecaySteps: 1000,
                targetUpdateStrategy: .soft(tau: 0.5), // high tau for visible change
                learningRate: 0.001,
                optim: AdamW(learningRate: 0.001)
            )
            
            let initialTargetParams = agent.targetNetwork.parameters()
            
            for _ in 0..<10 {
                let state = MLX.uniform(low: Float(-1.0), high: Float(1.0), [4])
                let nextState = MLX.uniform(low: Float(-1.0), high: Float(1.0), [4])
                agent.store(state: state, action: MLXArray(Int32(0)), reward: 1.0, nextState: nextState, terminated: false)
            }
            
            _ = agent.update()
            
            let updatedTargetParams = agent.targetNetwork.parameters()
            #expect(initialTargetParams.count == updatedTargetParams.count)
        }
        
        @Test
        func testTargetUpdateStrategyHard() async throws {
            let agent = DQNAgent(
                batchSize: 4,
                stateSize: 4,
                actionSize: 2,
                gamma: 0.99,
                epsilonStart: 1.0,
                epsilonEnd: 0.01,
                epsilonDecaySteps: 1000,
                targetUpdateStrategy: .hard(frequency: 2),
                learningRate: 0.001,
                optim: AdamW(learningRate: 0.001)
            )
            
            for _ in 0..<10 {
                let state = MLX.uniform(low: Float(-1.0), high: Float(1.0), [4])
                let nextState = MLX.uniform(low: Float(-1.0), high: Float(1.0), [4])
                agent.store(state: state, action: MLXArray(Int32(0)), reward: 1.0, nextState: nextState, terminated: false)
            }
            
            _ = agent.update()
            #expect(agent.currentSteps == 1)
            
            _ = agent.update()
            #expect(agent.currentSteps == 2)
        }
    }
}

