import Testing

@testable import Gymnazo

@Suite("AutoReset wrapper")
struct AutoResetWrapperTests {
    struct CounterEnv: Env {
        typealias Observation = Int
        typealias Action = Int

        let observation_space = Discrete(n: 100)
        let action_space = Discrete(n: 1)

        var spec: EnvSpec? = nil
        var render_mode: String? = nil

        var state: Int = 0

        mutating func step(_ action: Int) -> Step<Observation> {
            state += 1
            return Step(
                obs: state, reward: 1, terminated: true, truncated: false,
                info: ["step": .int(state)])
        }

        mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<Observation> {
            state = 0
            return Reset(obs: 0, info: ["reset": true])
        }
    }

    @Test
    func testSameStepAutoresetReturnsResetObservation() async throws {
        var env = CounterEnv().autoReset(mode: .sameStep)
        _ = env.reset(seed: nil, options: nil)
        let step = env.step(0)
        #expect(step.obs == 0)
        #expect(step.terminated)
        #expect(step.info["reset"]?.bool == true)
        #expect(step.final?.terminalObservation == 1)
        #expect(step.final?.terminalInfo["step"]?.int == 1)
    }

    @Test
    func testNextStepAutoresetResetsBeforeNextStep() async throws {
        var env = CounterEnv().autoReset(mode: .nextStep)
        _ = env.reset(seed: nil, options: nil)
        let step1 = env.step(0)
        #expect(step1.obs == 1)
        #expect(step1.terminated)
        #expect(step1.final?.terminalObservation == 1)
        let step2 = env.step(0)
        #expect(step2.obs == 1)
        #expect(step2.terminated)
    }

    @Test
    func testSameStepAutoResetTransitionDidReset() async throws {
        var env = CounterEnv().autoReset(mode: .sameStep)
        _ = env.reset(seed: nil, options: nil)
        let step = env.step(0)

        guard let final = step.final else {
            Issue.record("Expected final to be set on terminal step")
            return
        }

        #expect(final.terminalObservation == 1)

        switch final.autoReset {
        case .didReset(let observation, let info):
            #expect(observation == 0)
            #expect(info["reset"]?.bool == true)
        case .none, .willResetOnNextStep:
            Issue.record("Expected .didReset but got \(final.autoReset)")
        }
    }

    @Test
    func testNextStepAutoResetTransitionWillResetOnNextStep() async throws {
        var env = CounterEnv().autoReset(mode: .nextStep)
        _ = env.reset(seed: nil, options: nil)
        let step = env.step(0)

        guard let final = step.final else {
            Issue.record("Expected final to be set on terminal step")
            return
        }

        #expect(final.terminalObservation == 1)

        switch final.autoReset {
        case .willResetOnNextStep:
            break
        case .none, .didReset:
            Issue.record("Expected .willResetOnNextStep but got \(final.autoReset)")
        }
    }

    @Test
    func testDisabledModeNoFinal() async throws {
        var env = CounterEnv().autoReset(mode: .disabled)
        _ = env.reset(seed: nil, options: nil)
        let step = env.step(0)

        #expect(step.obs == 1)
        #expect(step.terminated)
        #expect(step.final == nil)
    }

    @Test
    func testTerminalObservationDiffersFromStepObsInSameStepMode() async throws {
        var env = CounterEnv().autoReset(mode: .sameStep)
        _ = env.reset(seed: nil, options: nil)
        let step = env.step(0)

        #expect(step.obs == 0)
        #expect(step.final?.terminalObservation == 1)
        #expect(step.obs != step.final?.terminalObservation)
    }
}
