import Testing
import MLX
@testable import Gymnazo

@Suite("Image Observation Wrappers")
struct ImageObservationTests {
    
    @Test
    func testGrayscaleOutputShape() async throws {
        var env = CarRacing()
        var grayEnv = GrayscaleObservation(env: env, keepDim: false)
        
        let result = grayEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [96, 96])
    }
    
    @Test
    func testGrayscaleKeepDimOutputShape() async throws {
        var env = CarRacing()
        var grayEnv = GrayscaleObservation(env: env, keepDim: true)
        
        let result = grayEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [96, 96, 1])
    }
    
    @Test
    func testGrayscaleObservationSpace() async throws {
        var env = CarRacing()
        var grayEnv = GrayscaleObservation(env: env, keepDim: false)
        
        #expect(grayEnv.observation_space.shape == [96, 96])
        #expect(grayEnv.observation_space.dtype == .uint8)
    }
    
    @Test
    func testGrayscaleStep() async throws {
        var env = CarRacing()
        var grayEnv = GrayscaleObservation(env: env, keepDim: false)
        
        _ = grayEnv.reset(seed: 42)
        let action = MLXArray([0.0, 0.5, 0.0] as [Float32])
        let result = grayEnv.step(action)
        
        #expect(result.obs.shape == [96, 96])
    }
    
    @Test
    func testResizeOutputShape() async throws {
        var env = CarRacing()
        var resizedEnv = ResizeObservation(env: env, shape: (84, 84))
        
        let result = resizedEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [84, 84, 3])
    }
    
    @Test
    func testResizeObservationSpace() async throws {
        var env = CarRacing()
        var resizedEnv = ResizeObservation(env: env, shape: (84, 84))
        
        #expect(resizedEnv.observation_space.shape == [84, 84, 3])
        #expect(resizedEnv.observation_space.dtype == .uint8)
    }
    
    @Test
    func testResizeStep() async throws {
        var env = CarRacing()
        var resizedEnv = ResizeObservation(env: env, shape: (84, 84))
        
        _ = resizedEnv.reset(seed: 42)
        let action = MLXArray([0.0, 0.5, 0.0] as [Float32])
        let result = resizedEnv.step(action)
        
        #expect(result.obs.shape == [84, 84, 3])
    }
    
    @Test
    func testResizeSmaller() async throws {
        var env = CarRacing()
        var resizedEnv = ResizeObservation(env: env, shape: (48, 48))
        
        let result = resizedEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [48, 48, 3])
    }
    
    @Test
    func testFrameStackOutputShape() async throws {
        var env = CarRacing()
        var stackedEnv = FrameStackObservation(env: env, stackSize: 4)
        
        let result = stackedEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [4, 96, 96, 3])
    }
    
    @Test
    func testFrameStackObservationSpace() async throws {
        var env = CarRacing()
        var stackedEnv = FrameStackObservation(env: env, stackSize: 4)
        
        #expect(stackedEnv.observation_space.shape == [4, 96, 96, 3])
        #expect(stackedEnv.observation_space.dtype == .uint8)
    }
    
    @Test
    func testFrameStackStep() async throws {
        var env = CarRacing()
        var stackedEnv = FrameStackObservation(env: env, stackSize: 4)
        
        _ = stackedEnv.reset(seed: 42)
        let action = MLXArray([0.0, 0.5, 0.0] as [Float32])
        let result = stackedEnv.step(action)
        
        #expect(result.obs.shape == [4, 96, 96, 3])
    }
    
    @Test
    func testFrameStackDifferentSize() async throws {
        var env = CarRacing()
        var stackedEnv = FrameStackObservation(env: env, stackSize: 2)
        
        let result = stackedEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [2, 96, 96, 3])
    }
    
    @Test
    func testGrayscaleThenResize() async throws {
        var env = CarRacing()
        var grayEnv = GrayscaleObservation(env: env, keepDim: true)
        var resizedEnv = ResizeObservation(env: grayEnv, shape: (84, 84))
        
        let result = resizedEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [84, 84, 1])
    }
    
    @Test
    func testGrayscaleThenFrameStack() async throws {
        var env = CarRacing()
        var grayEnv = GrayscaleObservation(env: env, keepDim: false)
        var stackedEnv = FrameStackObservation(env: grayEnv, stackSize: 4)
        
        let result = stackedEnv.reset(seed: 42)
        
        #expect(result.obs.shape == [4, 96, 96])
    }
}
