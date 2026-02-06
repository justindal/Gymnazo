import Testing
import MLX
@testable import Gymnazo

@Suite("CliffWalking Environment Tests")
struct CliffWalkingTests {
    func makeCliffWalking(isSlippery: Bool? = nil, renderMode: RenderMode? = nil) async throws -> CliffWalking {
        var options: EnvOptions = [:]
        if let isSlippery {
            options["is_slippery"] = isSlippery
        }
        if let renderMode {
            options["render_mode"] = renderMode.rawValue
        }
        let env = try await Gymnazo.make("CliffWalking", options: options)
        guard let cliffWalking = env.unwrapped as? CliffWalking else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "CliffWalking",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return cliffWalking
    }
    
    @Test("Reset always starts at state 36")
    func testResetStartsAtState36() async throws {
        let env = try await makeCliffWalking()
        
        for seed in [42, 123, 999, 0, 12345] as [UInt64] {
            let obs = try! env.reset(seed: seed).obs
            #expect(obs.item(Int.self) == CliffWalking.startState)
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
    func testActionSpace() async throws {
        let env = try await makeCliffWalking()

        guard let actionSpace = env.actionSpace as? Discrete else {
            Issue.record("Action space is not Discrete")
            return
        }

        #expect(actionSpace.n == 4)
        for action in 0..<4 {
            #expect(actionSpace.contains(MLXArray(Int32(action))))
        }
        #expect(!actionSpace.contains(MLXArray(Int32(-1))))
        #expect(!actionSpace.contains(MLXArray(Int32(4))))
    }
    
    @Test("Observation space contains valid states")
    func testObservationSpace() async throws {
        let env = try await makeCliffWalking()

        guard let observationSpace = env.observationSpace as? Discrete else {
            Issue.record("Observation space is not Discrete")
            return
        }

        #expect(observationSpace.n == 48)
        for state in 0..<48 {
            #expect(observationSpace.contains(MLXArray(Int32(state))))
        }
        #expect(!observationSpace.contains(MLXArray(Int32(-1))))
        #expect(!observationSpace.contains(MLXArray(Int32(48))))
    }
    
    @Test("Moving up from start goes to row 2")
    func testMoveUp() async throws {
        let env = try await makeCliffWalking()
        _ = try env.reset(seed: 42)
        
        let result = try env.step(MLXArray(Int32(0)))
        let obs = result.obs
        let reward = result.reward
        let terminated = result.terminated
        
        let expectedState = CliffWalking.positionToState(row: 2, col: 0)
        #expect(obs.item(Int.self) == expectedState)
        #expect(reward == -1.0)
        #expect(!terminated)
    }
    
    @Test("Moving left from start stays at start (wall collision)")
    func testMoveLeftFromStart() async throws {
        let env = try await makeCliffWalking()
        _ = try env.reset(seed: 42)
        
        let result = try env.step(MLXArray(Int32(3)))
        let obs = result.obs
        let reward = result.reward
        let terminated = result.terminated
        
        #expect(obs.item(Int.self) == CliffWalking.startState)
        #expect(reward == -1.0)
        #expect(!terminated)
    }
    
    @Test("Stepping on cliff returns to start with -100 reward")
    func testCliffPenalty() async throws {
        let env = try await makeCliffWalking()
        _ = try env.reset(seed: 42)
        
        let result = try env.step(MLXArray(Int32(1)))
        let obs = result.obs
        let reward = result.reward
        let terminated = result.terminated
        
        #expect(obs.item(Int.self) == CliffWalking.startState)
        #expect(reward == -100.0)
        #expect(!terminated)
    }
    
    @Test("Multiple cliff steps always return to start")
    func testMultipleCliffSteps() async throws {
        let env = try await makeCliffWalking()
        
        for _ in 0..<5 {
            _ = try env.reset(seed: 42)
            let result = try env.step(MLXArray(Int32(1)))
            let obs = result.obs
            let reward = result.reward
            #expect(obs.item(Int.self) == CliffWalking.startState)
            #expect(reward == -100.0)
        }
    }
    
    @Test("Path to goal terminates episode")
    func testPathToGoal() async throws {
        let env = try await makeCliffWalking()
        _ = try env.reset(seed: 42)
        
        _ = try env.step(MLXArray(Int32(0)))
        
        for _ in 0..<11 {
            let result = try env.step(MLXArray(Int32(1)))
            if result.terminated { break }
        }
        
        let result = try env.step(MLXArray(Int32(2)))
        let obs = result.obs
        let terminated = result.terminated
        
        #expect(obs.item(Int.self) == CliffWalking.goalState)
        #expect(terminated)
    }
    
    @Test("ANSI rendering includes grid symbols")
    func testAnsiRendering() async throws {
        let env = try await makeCliffWalking(renderMode: .ansi)
        _ = try env.reset(seed: 42)
        
        let output = env.renderAnsi()
        
        #expect(output.contains("x"))
        #expect(output.contains("G"))
        #expect(output.contains("C"))
    }
    
    @Test("Slippery mode creates stochastic transitions")
    func testSlipperyMode() async throws {
        let env = try await makeCliffWalking(isSlippery: true)
        
        var outcomes: Set<Int> = []
        for seed in 0..<100 as Range<UInt64> {
            _ = try env.reset(seed: seed)
            _ = try env.step(MLXArray(Int32(0)))
            let obs = try env.step(MLXArray(Int32(0))).obs
            outcomes.insert(obs.item(Int.self))
        }
        
        #expect(outcomes.count > 1)
    }
    
    @Test("Non-slippery mode is deterministic")
    func testNonSlipperyDeterministic() async throws {
        let env = try await makeCliffWalking(isSlippery: false)
        
        var outcomes: [Int] = []
        for seed in 0..<10 as Range<UInt64> {
            _ = try env.reset(seed: seed)
            outcomes.append(try env.step(MLXArray(Int32(0))).obs.item(Int.self))
        }
        
        let expected = CliffWalking.positionToState(row: 2, col: 0)
        for obs in outcomes {
            #expect(obs == expected)
        }
    }
    
    @Test("Wall collision on top edge")
    func testTopWallCollision() async throws {
        let env = try await makeCliffWalking()
        _ = try env.reset(seed: 42)
        
        _ = try env.step(MLXArray(Int32(0)))
        _ = try env.step(MLXArray(Int32(0)))
        let obs1 = try env.step(MLXArray(Int32(0))).obs
        
        #expect(obs1.item(Int.self) == CliffWalking.positionToState(row: 0, col: 0))
        
        let obs2 = try env.step(MLXArray(Int32(0))).obs
        #expect(obs2.item(Int.self) == CliffWalking.positionToState(row: 0, col: 0))
    }
    
    @Test("Wall collision on right edge")
    func testRightWallCollision() async throws {
        let env = try await makeCliffWalking()
        _ = try env.reset(seed: 42)
        
        _ = try env.step(MLXArray(Int32(0)))
        
        for _ in 0..<12 {
            _ = try env.step(MLXArray(Int32(1)))
        }
        
        let obs = try env.step(MLXArray(Int32(1))).obs
        #expect(obs.item(Int.self) == CliffWalking.positionToState(row: 2, col: 11))
    }
    
    @Test
    @MainActor
    func testGymnazoMakeCliffWalking() async throws {
        let env = try await Gymnazo.make("CliffWalking")
        let cliffWalking = env.unwrapped as! CliffWalking
        let obs = try cliffWalking.reset(seed: 123).obs
        #expect(obs.item(Int.self) == CliffWalking.startState)
    }
    
    @Test
    @MainActor
    func testGymnazoMakeCliffWalkingWithKwargs() async throws {
        let env = try await Gymnazo.make(
            "CliffWalking",
            options: ["render_mode": "ansi"]
        )
        let cliffWalking = env.unwrapped as! CliffWalking
        let obs = try cliffWalking.reset(seed: 42).obs
        #expect(obs.item(Int.self) == CliffWalking.startState)
    }
}

