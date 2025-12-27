//
// CliffWalkingTests.swift
//

import Testing
@testable import Gymnazo

@Suite("CliffWalking Environment Tests")
struct CliffWalkingTests {
    
    @Test("Reset always starts at state 36")
    func testResetStartsAtState36() {
        let env = CliffWalking()
        
        for seed in [42, 123, 999, 0, 12345] as [UInt64] {
            let (obs, _) = env.reset(seed: seed)
            #expect(obs == CliffWalking.startState)
        }
    }
    
    @Test("State and position conversion roundtrip")
    func testStatePositionConversion() {
        for state in 0..<48 {
            let (row, col) = CliffWalking.stateToPosition(state)
            let reconstructed = CliffWalking.positionToState(row: row, col: col)
            #expect(reconstructed == state)
        }
    }
    
    @Test("Position to state encoding")
    func testPositionToState() {
        #expect(CliffWalking.positionToState(row: 0, col: 0) == 0)
        #expect(CliffWalking.positionToState(row: 0, col: 11) == 11)
        #expect(CliffWalking.positionToState(row: 3, col: 0) == 36)
        #expect(CliffWalking.positionToState(row: 3, col: 11) == 47)
    }
    
    @Test("State to position decoding")
    func testStateToPosition() {
        #expect(CliffWalking.stateToPosition(0) == (0, 0))
        #expect(CliffWalking.stateToPosition(11) == (0, 11))
        #expect(CliffWalking.stateToPosition(36) == (3, 0))
        #expect(CliffWalking.stateToPosition(47) == (3, 11))
    }
    
    @Test("Action space contains valid actions")
    func testActionSpace() {
        let env = CliffWalking()
        
        #expect(env.action_space.n == 4)
        for action in 0..<4 {
            #expect(env.action_space.contains(action))
        }
        #expect(!env.action_space.contains(-1))
        #expect(!env.action_space.contains(4))
    }
    
    @Test("Observation space contains valid states")
    func testObservationSpace() {
        let env = CliffWalking()
        
        #expect(env.observation_space.n == 48)
        for state in 0..<48 {
            #expect(env.observation_space.contains(state))
        }
        #expect(!env.observation_space.contains(-1))
        #expect(!env.observation_space.contains(48))
    }
    
    @Test("Moving up from start goes to row 2")
    func testMoveUp() {
        let env = CliffWalking()
        _ = env.reset(seed: 42)
        
        let (obs, reward, terminated, _, _) = env.step(0)
        
        let expectedState = CliffWalking.positionToState(row: 2, col: 0)
        #expect(obs == expectedState)
        #expect(reward == -1.0)
        #expect(!terminated)
    }
    
    @Test("Moving left from start stays at start (wall collision)")
    func testMoveLeftFromStart() {
        let env = CliffWalking()
        _ = env.reset(seed: 42)
        
        let (obs, reward, terminated, _, _) = env.step(3)
        
        #expect(obs == CliffWalking.startState)
        #expect(reward == -1.0)
        #expect(!terminated)
    }
    
    @Test("Stepping on cliff returns to start with -100 reward")
    func testCliffPenalty() {
        let env = CliffWalking()
        _ = env.reset(seed: 42)
        
        let (obs, reward, terminated, _, _) = env.step(1)
        
        #expect(obs == CliffWalking.startState)
        #expect(reward == -100.0)
        #expect(!terminated)
    }
    
    @Test("Multiple cliff steps always return to start")
    func testMultipleCliffSteps() {
        let env = CliffWalking()
        
        for _ in 0..<5 {
            _ = env.reset(seed: 42)
            let (obs, reward, _, _, _) = env.step(1)
            #expect(obs == CliffWalking.startState)
            #expect(reward == -100.0)
        }
    }
    
    @Test("Path to goal terminates episode")
    func testPathToGoal() {
        let env = CliffWalking()
        _ = env.reset(seed: 42)
        
        _ = env.step(0)
        
        for _ in 0..<11 {
            let (_, _, terminated, _, _) = env.step(1)
            if terminated { break }
        }
        
        let (obs, _, terminated, _, _) = env.step(2)
        
        #expect(obs == CliffWalking.goalState)
        #expect(terminated)
    }
    
    @Test("ANSI rendering includes grid symbols")
    func testAnsiRendering() {
        let env = CliffWalking(render_mode: "ansi")
        _ = env.reset(seed: 42)
        
        let output = env.renderAnsi()
        
        #expect(output.contains("x"))
        #expect(output.contains("G"))
        #expect(output.contains("C"))
    }
    
    @Test("Slippery mode creates stochastic transitions")
    func testSlipperyMode() {
        let env = CliffWalking(isSlippery: true)
        
        var outcomes: Set<Int> = []
        for seed in 0..<100 as Range<UInt64> {
            _ = env.reset(seed: seed)
            _ = env.step(0)
            let (obs, _, _, _, _) = env.step(0)
            outcomes.insert(obs)
        }
        
        #expect(outcomes.count > 1)
    }
    
    @Test("Non-slippery mode is deterministic")
    func testNonSlipperyDeterministic() {
        let env = CliffWalking(isSlippery: false)
        
        var outcomes: [Int] = []
        for seed in 0..<10 as Range<UInt64> {
            _ = env.reset(seed: seed)
            let (obs, _, _, _, _) = env.step(0)
            outcomes.append(obs)
        }
        
        let expected = CliffWalking.positionToState(row: 2, col: 0)
        for obs in outcomes {
            #expect(obs == expected)
        }
    }
    
    @Test("Wall collision on top edge")
    func testTopWallCollision() {
        let env = CliffWalking()
        _ = env.reset(seed: 42)
        
        _ = env.step(0)
        _ = env.step(0)
        let (obs1, _, _, _, _) = env.step(0)
        
        #expect(obs1 == CliffWalking.positionToState(row: 0, col: 0))
        
        let (obs2, _, _, _, _) = env.step(0)
        #expect(obs2 == CliffWalking.positionToState(row: 0, col: 0))
    }
    
    @Test("Wall collision on right edge")
    func testRightWallCollision() {
        let env = CliffWalking()
        _ = env.reset(seed: 42)
        
        _ = env.step(0)
        
        for _ in 0..<12 {
            _ = env.step(1)
        }
        
        let (obs, _, _, _, _) = env.step(1)
        #expect(obs == CliffWalking.positionToState(row: 2, col: 11))
    }
    
    @Test
    @MainActor
    func testGymnazoMakeCliffWalking() async throws {
        let env = Gymnazo.make("CliffWalking")
        let cliffWalking = env.unwrapped as! CliffWalking
        let (obs, _) = cliffWalking.reset(seed: 123)
        #expect(obs == CliffWalking.startState)
    }
    
    @Test
    @MainActor
    func testGymnazoMakeCliffWalkingWithKwargs() async throws {
        let env = Gymnazo.make("CliffWalking", kwargs: ["render_mode": "ansi"])
        let cliffWalking = env.unwrapped as! CliffWalking
        let (obs, _) = cliffWalking.reset(seed: 42)
        #expect(obs == CliffWalking.startState)
    }
}

