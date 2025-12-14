import MLX

public struct FlattenObservation<InnerEnv: Env>: TransformingWrapper {
    public typealias InnerEnv = InnerEnv
    public typealias Observation = MLXArray
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = Box
    public typealias ActionSpace = InnerEnv.ActionSpace

    public var env: InnerEnv
    public let observation_space: Box

    private let baseSpace: any Space

    public init(env: InnerEnv) {
        self.env = env
        let s: any Space = env.observation_space
        self.baseSpace = s
        guard let box = flattenSpaceToBox(s) else {
            fatalError("FlattenObservation requires an observation space that flattens to Box")
        }
        self.observation_space = box
    }

    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> ResetResult {
        let result = env.reset(seed: seed, options: options)
        let flattened = flatten(space: baseSpace, sample: result.obs)
        guard let flatObs = flattened as? MLXArray else {
            fatalError("FlattenObservation expects MLXArray output from flatten(space:sample:)")
        }
        return (obs: flatObs, info: result.info)
    }

    public mutating func step(_ action: InnerEnv.Action) -> StepResult {
        let result = env.step(action)
        let flattened = flatten(space: baseSpace, sample: result.obs)
        guard let flatObs = flattened as? MLXArray else {
            fatalError("FlattenObservation expects MLXArray output from flatten(space:sample:)")
        }
        return (
            obs: flatObs,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
}

