import Testing
import MLX
@testable import ExploreRLCore

/// minimal continuous env whose step returns the action it received as the observation.
final class DummyBoxEnv: Environment {
    typealias Observation = MLXArray
    typealias Action = MLXArray
    typealias ObservationSpace = Box
    typealias ActionSpace = Box

    let action_space: Box
    let observation_space: Box
    var spec: EnvSpec? = nil
    var render_mode: String? = nil

    init(low: Float = -2.0, high: Float = 2.0, shape: [Int] = [2]) {
        self.action_space = Box(low: low, high: high, shape: shape, dtype: .float32)
        self.observation_space = Box(low: low, high: high, shape: shape, dtype: .float32)
    }

    func step(_ action: MLXArray) -> (obs: MLXArray, reward: Double, terminated: Bool, truncated: Bool, info: [String : Any]) {
        return (obs: action, reward: 0.0, terminated: false, truncated: false, info: [:])
    }

    func reset(seed: UInt64?, options: [String : Any]?) -> (obs: MLXArray, info: [String : Any]) {
        let zeros = MLXArray.zeros(action_space.shape ?? [1], type: Float.self)
        return (obs: zeros, info: [:])
    }
}

@Suite("Action wrappers on Box spaces")
struct ActionWrappersTests {
    @Test
    func testClipAction() async throws {
        let env = DummyBoxEnv(low: -1.0, high: 1.0, shape: [3])
        var wrapper = ClipAction(env: env)
        _ = wrapper.reset(seed: 0, options: nil)
        let action = MLXArray([2.0 as Float, -2.0 as Float, 0.5 as Float])
        let result = wrapper.step(action)
        let obs = result.obs.asArray(Float.self)
        #expect(abs(obs[0] - 1.0) < 1e-6)
        #expect(abs(obs[1] - (-1.0)) < 1e-6)
        #expect(abs(obs[2] - 0.5) < 1e-6)
    }

    @Test
    func testRescaleAction() async throws {
        let env = DummyBoxEnv(low: 0.0, high: 2.0, shape: [2])
        var wrapper = RescaleAction(env: env, sourceLow: -1.0, sourceHigh: 1.0)
        _ = wrapper.reset(seed: 0, options: nil)
        
        // input in [-1,1]; expect mapping to [0,2]
        let action = MLXArray([-1.0 as Float, 1.0 as Float])
        let result = wrapper.step(action)
        let obs = result.obs.asArray(Float.self)
        #expect(abs(obs[0] - 0.0) < 1e-6)
        #expect(abs(obs[1] - 2.0) < 1e-6)
    }
}

