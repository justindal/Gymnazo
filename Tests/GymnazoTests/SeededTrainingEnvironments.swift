import MLX

@testable import Gymnazo

struct SeededDiscreteTrainingEnvironment: Env {
    let observationSpace: any Space = Box(low: -1, high: 1, shape: [4])
    let actionSpace: any Space = Discrete(n: 2)
    var spec: EnvSpec?
    var renderMode: RenderMode?
    private var stepCount = 0

    mutating func reset(seed _: UInt64?, options _: EnvOptions?) throws -> Reset {
        stepCount = 0
        return Reset(obs: MLXArray([0, 0, 0, 0] as [Float]))
    }

    mutating func step(_: MLXArray) throws -> Step {
        stepCount += 1
        let terminated = stepCount == 3
        return Step(
            obs: MLXArray([Float(stepCount), 0, 0, 0]),
            reward: 1,
            terminated: terminated,
            truncated: false
        )
    }
}

struct SeededContinuousTrainingEnvironment: Env {
    let observationSpace: any Space = Box(low: -1, high: 1, shape: [3])
    let actionSpace: any Space = Box(low: -1, high: 1, shape: [1])
    var spec: EnvSpec?
    var renderMode: RenderMode?
    private var stepCount = 0

    mutating func reset(seed _: UInt64?, options _: EnvOptions?) throws -> Reset {
        stepCount = 0
        return Reset(obs: MLXArray([0, 0, 0] as [Float]))
    }

    mutating func step(_: MLXArray) throws -> Step {
        stepCount += 1
        let terminated = stepCount == 3
        return Step(
            obs: MLXArray([Float(stepCount), 0, 0]),
            reward: 1,
            terminated: terminated,
            truncated: false
        )
    }
}
