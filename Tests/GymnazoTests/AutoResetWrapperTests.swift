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
            return Step(obs: state, reward: 1, terminated: true, truncated: false, info: ["step": .int(state)])
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
        #expect(step.final?.obs == 1)
        #expect(step.final?.info["step"]?.int == 1)
    }

    @Test
    func testNextStepAutoresetResetsBeforeNextStep() async throws {
        var env = CounterEnv().autoReset(mode: .nextStep)
        _ = env.reset(seed: nil, options: nil)
        let step1 = env.step(0)
        #expect(step1.obs == 1)
        #expect(step1.terminated)
        #expect(step1.final?.obs == 1)
        let step2 = env.step(0)
        #expect(step2.obs == 1)
        #expect(step2.terminated)
    }
}

