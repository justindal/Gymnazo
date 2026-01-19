import Testing
import MLX
@testable import Gymnazo

@Suite("Image Observation Wrappers")
struct ImageObservationTests {
    func makeCarRacing() async throws -> CarRacing {
        let env: AnyEnv<MLXArray, MLXArray> = try await Gymnazo.make("CarRacing")
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
        var env = try await makeCarRacing()
        var grayEnv = try GrayscaleObservation(env: env, keepDim: false)
        
        let result = try grayEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [96, 96])
    }
    
    @Test
    func testGrayscaleKeepDimOutputShape() async throws {
        var env = try await makeCarRacing()
        var grayEnv = try GrayscaleObservation(env: env, keepDim: true)
        
        let result = try grayEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [96, 96, 1])
    }
    
    @Test
    func testGrayscaleObservationSpace() async throws {
        var env = try await makeCarRacing()
        var grayEnv = try GrayscaleObservation(env: env, keepDim: false)
        
        #expect(grayEnv.observationSpace.shape == [96, 96])
        #expect(grayEnv.observationSpace.dtype == .uint8)
    }
    
    @Test
    func testGrayscaleStep() async throws {
        var env = try await makeCarRacing()
        var grayEnv = try GrayscaleObservation(env: env, keepDim: false)
        
        _ = try grayEnv.reset(seed: 42)
        let action = MLXArray([0.0, 0.5, 0.0] as [Float32])
        let result = try grayEnv.step(action)
        
        #expect(result.obs.shape == [96, 96])
    }
    
    @Test
    func testResizeOutputShape() async throws {
        var env = try await makeCarRacing()
        var resizedEnv = try ResizeObservation(env: env, shape: (84, 84))
        
        let result = try resizedEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [84, 84, 3])
    }
    
    @Test
    func testResizeObservationSpace() async throws {
        var env = try await makeCarRacing()
        var resizedEnv = try ResizeObservation(env: env, shape: (84, 84))
        
        #expect(resizedEnv.observationSpace.shape == [84, 84, 3])
        #expect(resizedEnv.observationSpace.dtype == .uint8)
    }
    
    @Test
    func testResizeStep() async throws {
        var env = try await makeCarRacing()
        var resizedEnv = try ResizeObservation(env: env, shape: (84, 84))
        
        _ = try resizedEnv.reset(seed: 42)
        let action = MLXArray([0.0, 0.5, 0.0] as [Float32])
        let result = try resizedEnv.step(action)
        
        #expect(result.obs.shape == [84, 84, 3])
    }
    
    @Test
    func testResizeSmaller() async throws {
        var env = try await makeCarRacing()
        var resizedEnv = try ResizeObservation(env: env, shape: (48, 48))
        
        let result = try resizedEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [48, 48, 3])
    }
    
    @Test
    func testFrameStackOutputShape() async throws {
        var env = try await makeCarRacing()
        var stackedEnv = try FrameStackObservation(env: env, stackSize: 4)
        
        let result = try stackedEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [4, 96, 96, 3])
    }
    
    @Test
    func testFrameStackObservationSpace() async throws {
        var env = try await makeCarRacing()
        var stackedEnv = try FrameStackObservation(env: env, stackSize: 4)
        
        #expect(stackedEnv.observationSpace.shape == [4, 96, 96, 3])
        #expect(stackedEnv.observationSpace.dtype == .uint8)
    }
    
    @Test
    func testFrameStackStep() async throws {
        var env = try await makeCarRacing()
        var stackedEnv = try FrameStackObservation(env: env, stackSize: 4)
        
        _ = try stackedEnv.reset(seed: 42)
        let action = MLXArray([0.0, 0.5, 0.0] as [Float32])
        let result = try stackedEnv.step(action)
        
        #expect(result.obs.shape == [4, 96, 96, 3])
    }
    
    @Test
    func testFrameStackDifferentSize() async throws {
        var env = try await makeCarRacing()
        var stackedEnv = try FrameStackObservation(env: env, stackSize: 2)
        
        let result = try stackedEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [2, 96, 96, 3])
    }
    
    @Test
    func testGrayscaleThenResize() async throws {
        var env = try await makeCarRacing()
        var grayEnv = try GrayscaleObservation(env: env, keepDim: true)
        var resizedEnv = try ResizeObservation(env: grayEnv, shape: (84, 84))
        
        let result = try resizedEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [84, 84, 1])
    }
    
    @Test
    func testGrayscaleThenFrameStack() async throws {
        var env = try await makeCarRacing()
        var grayEnv = try GrayscaleObservation(env: env, keepDim: false)
        var stackedEnv = try FrameStackObservation(env: grayEnv, stackSize: 4)
        
        let result = try stackedEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [4, 96, 96])
    }
}
