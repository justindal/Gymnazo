import MLX
import Testing

@testable import Gymnazo

@Suite("Classic control determinism", .serialized)
struct ClassicControlDeterminismTests {
    @Test
    func cartPoleUsesSeededSwiftState() throws {
        var first = CartPole()
        var second = CartPole()

        let firstReset = try first.reset(seed: 91)
        let secondReset = try second.reset(seed: 91)
        #expect(firstReset.obs.asArray(Float.self) == secondReset.obs.asArray(Float.self))

        for action in [0, 1, 1, 0, 1] {
            let firstStep = try first.step(MLXArray(Int32(action)))
            let secondStep = try second.step(MLXArray(Int32(action)))
            #expect(firstStep.obs.asArray(Float.self) == secondStep.obs.asArray(Float.self))
            #expect(firstStep.reward == secondStep.reward)
            #expect(firstStep.terminated == secondStep.terminated)
        }
    }

    @Test
    func mountainCarUsesSeededSwiftState() throws {
        var first = MountainCar()
        var second = MountainCar()

        let firstReset = try first.reset(seed: 23)
        let secondReset = try second.reset(seed: 23)
        #expect(firstReset.obs.asArray(Float.self) == secondReset.obs.asArray(Float.self))

        for action in [2, 0, 1, 2, 2] {
            let firstStep = try first.step(MLXArray(Int32(action)))
            let secondStep = try second.step(MLXArray(Int32(action)))
            #expect(firstStep.obs.asArray(Float.self) == secondStep.obs.asArray(Float.self))
            #expect(firstStep.reward == secondStep.reward)
            #expect(firstStep.terminated == secondStep.terminated)
        }
    }
}
