import MLX
import Testing

@testable import Gymnazo

private struct ImageBoundsEnvironment: Env {
    let actionSpace: any Space = Discrete(n: 1)
    let observationSpace: any Space
    var observation: MLXArray
    var spec: EnvSpec?
    var renderMode: RenderMode?

    init(low: MLXArray, high: MLXArray, observation: MLXArray) {
        observationSpace = Box(low: low, high: high)
        self.observation = observation
    }

    mutating func step(_ action: MLXArray) throws -> Step {
        Step(obs: observation, reward: 0, terminated: false, truncated: false)
    }

    mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset {
        Reset(obs: observation)
    }
}

@Suite("Image Observation Wrappers")
struct ImageObservationTests {
    func makeCarRacing() async throws -> CarRacing {
        let env = try await Gymnazo.make("CarRacing")
        guard let carRacing = env.unwrapped as? CarRacing else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "CarRacing",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return carRacing
    }

    @Test
    func testGrayscaleOutputShape() async throws {
        let env = try await makeCarRacing()
        var grayEnv = try GrayscaleObservation(env: env, keepDim: false)

        let result = try grayEnv.reset(seed: 42)

        #expect(result.obs.shape == [96, 96])
    }

    @Test
    func testGrayscaleKeepDimOutputShape() async throws {
        let env = try await makeCarRacing()
        var grayEnv = try GrayscaleObservation(env: env, keepDim: true)

        let result = try grayEnv.reset(seed: 42)

        #expect(result.obs.shape == [96, 96, 1])
    }

    @Test
    func testGrayscaleObservationSpace() async throws {
        let env = try await makeCarRacing()
        let grayEnv = try GrayscaleObservation(env: env, keepDim: false)

        #expect(grayEnv.observationSpace.shape == [96, 96])
        #expect(grayEnv.observationSpace.dtype == .uint8)
    }

    @Test
    func testGrayscaleStep() async throws {
        let env = try await makeCarRacing()
        var grayEnv = try GrayscaleObservation(env: env, keepDim: false)

        _ = try grayEnv.reset(seed: 42)
        let action = MLXArray([0.0, 0.5, 0.0] as [Float32])
        let result = try grayEnv.step(action)

        #expect(result.obs.shape == [96, 96])
    }

    @Test
    func testResizeOutputShape() async throws {
        let env = try await makeCarRacing()
        var resizedEnv = try ResizeObservation(env: env, shape: (84, 84))

        let result = try resizedEnv.reset(seed: 42)

        #expect(result.obs.shape == [84, 84, 3])
    }

    @Test
    func testResizeObservationSpace() async throws {
        let env = try await makeCarRacing()
        let resizedEnv = try ResizeObservation(env: env, shape: (84, 84))

        #expect(resizedEnv.observationSpace.shape == [84, 84, 3])
        #expect(resizedEnv.observationSpace.dtype == .uint8)
    }

    @Test
    func testResizeStep() async throws {
        let env = try await makeCarRacing()
        var resizedEnv = try ResizeObservation(env: env, shape: (84, 84))

        _ = try resizedEnv.reset(seed: 42)
        let action = MLXArray([0.0, 0.5, 0.0] as [Float32])
        let result = try resizedEnv.step(action)

        #expect(result.obs.shape == [84, 84, 3])
    }

    @Test
    func testResizeSmaller() async throws {
        let env = try await makeCarRacing()
        var resizedEnv = try ResizeObservation(env: env, shape: (48, 48))

        let result = try resizedEnv.reset(seed: 42)

        #expect(result.obs.shape == [48, 48, 3])
    }

    @Test
    func testFrameStackOutputShape() async throws {
        let env = try await makeCarRacing()
        var stackedEnv = try FrameStackObservation(env: env, stackSize: 4)

        let result = try stackedEnv.reset(seed: 42)

        #expect(result.obs.shape == [4, 96, 96, 3])
    }

    @Test
    func testFrameStackObservationSpace() async throws {
        let env = try await makeCarRacing()
        let stackedEnv = try FrameStackObservation(env: env, stackSize: 4)

        #expect(stackedEnv.observationSpace.shape == [4, 96, 96, 3])
        #expect(stackedEnv.observationSpace.dtype == .uint8)
    }

    @Test
    func testFrameStackStep() async throws {
        let env = try await makeCarRacing()
        var stackedEnv = try FrameStackObservation(env: env, stackSize: 4)

        _ = try stackedEnv.reset(seed: 42)
        let action = MLXArray([0.0, 0.5, 0.0] as [Float32])
        let result = try stackedEnv.step(action)

        #expect(result.obs.shape == [4, 96, 96, 3])
    }

    @Test
    func testFrameStackDifferentSize() async throws {
        let env = try await makeCarRacing()
        var stackedEnv = try FrameStackObservation(env: env, stackSize: 2)

        let result = try stackedEnv.reset(seed: 42)

        #expect(result.obs.shape == [2, 96, 96, 3])
    }

    @Test
    func testGrayscaleThenResize() async throws {
        let env = try await makeCarRacing()
        let grayEnv = try GrayscaleObservation(env: env, keepDim: true)
        var resizedEnv = try ResizeObservation(env: grayEnv, shape: (84, 84))

        let result = try resizedEnv.reset(seed: 42)

        #expect(result.obs.shape == [84, 84, 1])
    }

    @Test
    func testGrayscaleThenFrameStack() async throws {
        let env = try await makeCarRacing()
        let grayEnv = try GrayscaleObservation(env: env, keepDim: false)
        var stackedEnv = try FrameStackObservation(env: grayEnv, stackSize: 4)

        let result = try stackedEnv.reset(seed: 42)

        #expect(result.obs.shape == [4, 96, 96])
    }

    @Test
    func grayscaleDerivesSignedFloatingBounds() throws {
        let low = MLXArray([Float(-10), Float(0), Float(10)]).reshaped([1, 1, 3])
        let high = MLXArray([Float(20), Float(30), Float(40)]).reshaped([1, 1, 3])
        let observation = MLXArray([Float(5), Float(10), Float(20)]).reshaped([1, 1, 3])
        let env = ImageBoundsEnvironment(low: low, high: high, observation: observation)
        var wrapper = try GrayscaleObservation(env: env)

        let reset = try wrapper.reset(seed: 1)
        let box = try #require(wrapper.observationSpace as? Box)
        eval(box.low, box.high, reset.obs)
        #expect(abs(box.low.item(Float.self) - (-1.85)) < 0.0001)
        #expect(abs(box.high.item(Float.self) - 28.15) < 0.0001)
        #expect(box.contains(reset.obs))
    }

    @Test
    func resizeDerivesPerPositionBounds() throws {
        let low = MLXArray([Float(-4), Float(-3), Float(-2), Float(-1)]).reshaped([2, 2, 1])
        let high = MLXArray([Float(1), Float(2), Float(3), Float(4)]).reshaped([2, 2, 1])
        let observation = MLXArray([Float(0), Float(1), Float(2), Float(3)]).reshaped([2, 2, 1])
        let env = ImageBoundsEnvironment(low: low, high: high, observation: observation)
        var wrapper = try ResizeObservation(env: env, shape: (4, 4))

        let reset = try wrapper.reset(seed: 1)
        let box = try #require(wrapper.observationSpace as? Box)
        eval(box.low, box.high, reset.obs)
        #expect(box.low[0, 0, 0].item(Float.self) == -4)
        #expect(box.low[3, 3, 0].item(Float.self) == -1)
        #expect(box.high[0, 0, 0].item(Float.self) == 1)
        #expect(box.high[3, 3, 0].item(Float.self) == 4)
        #expect(box.contains(reset.obs))
    }
}
