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

        mutating func step(_ action: Int) -> StepResult {
            state += 1
            return (
                obs: state,
                reward: 1,
                terminated: true,
                truncated: false,
                info: ["step": state]
            )
        }

        mutating func reset(seed: UInt64?, options: [String: Any]?) -> ResetResult {
            state = 0
            return (obs: 0, info: ["reset": true])
        }
    }

    @Test
    func testSameStepAutoresetReturnsResetObservation() async throws {
        var env = CounterEnv().autoReset(mode: .sameStep)
        _ = env.reset(seed: nil, options: nil)
        let step = env.step(0)
        #expect(step.obs == 0)
        #expect(step.terminated)
        #expect((step.info["reset"] as? Bool) == true)
        #expect((step.info["final_observation"] as? Int) == 1)
        let finalInfo = step.info["final_info"] as? [String: Any]
        #expect((finalInfo?["step"] as? Int) == 1)
    }

    @Test
    func testNextStepAutoresetResetsBeforeNextStep() async throws {
        var env = CounterEnv().autoReset(mode: .nextStep)
        _ = env.reset(seed: nil, options: nil)
        let step1 = env.step(0)
        #expect(step1.obs == 1)
        #expect(step1.terminated)
        #expect((step1.info["final_observation"] as? Int) == 1)
        let step2 = env.step(0)
        #expect(step2.obs == 1)
        #expect(step2.terminated)
    }
}

