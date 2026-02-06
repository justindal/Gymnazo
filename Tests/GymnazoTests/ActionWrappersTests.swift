import Testing
import MLX
@testable import Gymnazo

final class DummyBoxEnv: Env {
    let actionSpace: any Space
    let observationSpace: any Space
    var spec: EnvSpec? = nil
    var renderMode: RenderMode? = nil

    init(low: Float = -2.0, high: Float = 2.0, shape: [Int] = [2]) {
        self.actionSpace = Box(low: low, high: high, shape: shape, dtype: .float32)
        self.observationSpace = Box(low: low, high: high, shape: shape, dtype: .float32)
    }

    func step(_ action: MLXArray) throws -> Step {
        return Step(obs: action, reward: 0.0, terminated: false, truncated: false, info: [:])
    }

    func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        let zeros = MLXArray.zeros(actionSpace.shape ?? [1], type: Float.self)
        return Reset(obs: zeros, info: [:])
    }
}

@Suite("Action wrappers on Box spaces")
struct ActionWrappersTests {
    @Test
    func testClipAction() async throws {
        let env = DummyBoxEnv(low: -1.0, high: 1.0, shape: [3])
        var wrapper = ClipAction(env: env)
        _ = try wrapper.reset(seed: 0, options: nil)
        let action = MLXArray([2.0 as Float, -2.0 as Float, 0.5 as Float])
        let result = try wrapper.step(action)
        let obs = result.obs.asArray(Float.self)
        #expect(abs(obs[0] - 1.0) < 1e-6)
        #expect(abs(obs[1] - (-1.0)) < 1e-6)
        #expect(abs(obs[2] - 0.5) < 1e-6)
    }

    @Test
    func testRescaleAction() async throws {
        let env = DummyBoxEnv(low: 0.0, high: 2.0, shape: [2])
        var wrapper = RescaleAction(env: env, sourceLow: -1.0, sourceHigh: 1.0)
        _ = try wrapper.reset(seed: 0, options: nil)
        
        let action = MLXArray([-1.0 as Float, 1.0 as Float])
        let result = try wrapper.step(action)
        let obs = result.obs.asArray(Float.self)
        #expect(abs(obs[0] - 0.0) < 1e-6)
        #expect(abs(obs[1] - 2.0) < 1e-6)
    }
}
