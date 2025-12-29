import MLX

/// An observation wrapper that flattens observations into a 1D tensor.
///
/// This wrapper only supports environments whose observation space flattens to a ``Box`` via
///   ``flatten_space(_:)``. In particular, ``SequenceSpace`` and ``Graph`` preserve structure under `flatten_space`.
public struct FlattenObservation<InnerEnv: Env>: TransformingWrapper {
    public typealias InnerEnv = InnerEnv
    public typealias Observation = MLXArray
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = Box
    public typealias ActionSpace = InnerEnv.ActionSpace

    public var env: InnerEnv
    public let observation_space: Box

    private let baseSpace: any Space

    /// Creates the wrapper and computes the flattened observation space.
    public init(env: InnerEnv) {
        self.env = env
        let s: any Space = env.observation_space
        self.baseSpace = s
        guard let box = flattenSpaceToBox(s) else {
            fatalError("FlattenObservation requires an observation space that flattens to Box")
        }
        self.observation_space = box
    }

    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<Observation> {
        let result = env.reset(seed: seed, options: options)
        let flattened = flatten(space: baseSpace, sample: result.obs)
        guard let flatObs = flattened as? MLXArray else {
            fatalError("FlattenObservation expects MLXArray output from flatten(space:sample:)")
        }
        return Reset(obs: flatObs, info: result.info)
    }

    public mutating func step(_ action: InnerEnv.Action) -> Step<Observation> {
        let result = env.step(action)
        let flattened = flatten(space: baseSpace, sample: result.obs)
        guard let flatObs = flattened as? MLXArray else {
            fatalError("FlattenObservation expects MLXArray output from flatten(space:sample:)")
        }
        return Step(
            obs: flatObs,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }
}

