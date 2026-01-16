import MLX

/// An observation wrapper that flattens observations into a 1D tensor.
///
/// This wrapper only supports environments whose observation space flattens to a ``Box`` via
///   ``flatten_space(_:)``. In particular, ``SequenceSpace`` and ``Graph`` preserve structure under `flatten_space`.
public struct FlattenObservation<BaseEnv: Env>: TransformingWrapper {
    public var env: BaseEnv
    public let observationSpace: Box
    public var actionSpace: BaseEnv.ActionSpace { env.actionSpace }

    private let baseSpace: any Space

    /// Creates the wrapper and computes the flattened observation space.
    public init(env: BaseEnv) {
        self.env = env
        let s: any Space = env.observationSpace
        self.baseSpace = s
        guard let box = flattenSpaceToBox(s) else {
            fatalError("FlattenObservation requires an observation space that flattens to Box")
        }
        self.observationSpace = box
    }

    public mutating func reset(seed: UInt64?, options: [String: Any]?) -> Reset<MLXArray> {
        let result = env.reset(seed: seed, options: options)
        let flattened = flatten(space: baseSpace, sample: result.obs)
        guard let flatObs = flattened as? MLXArray else {
            fatalError("FlattenObservation expects MLXArray output from flatten(space:sample:)")
        }
        return Reset(obs: flatObs, info: result.info)
    }

    public mutating func step(_ action: BaseEnv.Action) -> Step<MLXArray> {
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
