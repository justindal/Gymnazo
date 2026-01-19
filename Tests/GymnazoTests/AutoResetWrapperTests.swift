import Testing

@testable import Gymnazo

@Suite("AutoReset wrapper")
struct AutoResetWrapperTests {
    struct CounterEnv: Env {
        typealias Observation = Int
        typealias Action = Int
        let observationSpace: any Space<Observation> = Discrete(n: 100)
        let actionSpace: any Space<Action> = Discrete(n: 1)

        var spec: EnvSpec? = nil
        var renderMode: RenderMode? = nil

        var state: Int = 0

        mutating func step(_ action: Int) throws -> Step<Observation> {
            state += 1
            return Step(
                obs: state, reward: 1, terminated: true, truncated: false,
                info: ["step": .int(state)])
        }

        mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset<Observation> {
            state = 0
            return Reset(obs: 0, info: ["reset": true])
        }
    }

    @Test
    func testSameStepAutoresetReturnsResetObservation() async throws {
        var env = CounterEnv().autoReset(mode: .sameStep)
        _ = try env.reset(seed: nil, options: nil)
        let step = try env.step(0)
        #expect(step.obs == 0)
        #expect(step.terminated)
        #expect(step.info["reset"]?.bool == true)
        #expect(step.info["final_observation"]?.cast(Int.self) == 1)
        #expect(step.info["final_info"]?.object?["step"]?.int == 1)
    }

    @Test
    func testNextStepAutoresetResetsBeforeNextStep() async throws {
        var env = CounterEnv().autoReset(mode: .nextStep)
        _ = try env.reset(seed: nil, options: nil)
        let step1 = try env.step(0)
        #expect(step1.obs == 1)
        #expect(step1.terminated)
        #expect(step1.info["final_observation"]?.cast(Int.self) == 1)
        let step2 = try env.step(0)
        #expect(step2.obs == 1)
        #expect(step2.terminated)
    }

    @Test
    func testDisabledModeNoFinal() async throws {
        var env = CounterEnv().autoReset(mode: .disabled)
        _ = try env.reset(seed: nil, options: nil)
        let step = try env.step(0)

        #expect(step.obs == 1)
        #expect(step.terminated)
        #expect(step.info["final_observation"] == nil)
    }

    @Test
    func testTerminalObservationDiffersFromStepObsInSameStepMode() async throws {
        var env = CounterEnv().autoReset(mode: .sameStep)
        _ = try env.reset(seed: nil, options: nil)
        let step = try env.step(0)

        #expect(step.obs == 0)
        #expect(step.info["final_observation"]?.cast(Int.self) == 1)
        #expect(step.info["final_observation"]?.cast(Int.self) != step.obs)
    }
}
