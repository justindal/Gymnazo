import MLX

/// An observation wrapper that flattens observations into a 1D tensor.
///
/// This wrapper only supports environments whose observation space flattens to a ``Box`` via
///   ``flatten_space(_:)``. In particular, ``SequenceSpace`` and ``Graph`` preserve structure under `flatten_space`.
public struct FlattenObservation<BaseEnv: Env>: TransformingWrapper {
    public typealias Observation = MLXArray
    public typealias Action = BaseEnv.Action

    public var env: BaseEnv
    public let observationSpace: any Space<Observation>
    public var actionSpace: any Space<BaseEnv.Action> { env.actionSpace }

    private let baseSpace: any Space<BaseEnv.Observation>

    /// Creates the wrapper and computes the flattened observation space.
    public init(env: BaseEnv) throws {
        self.env = env
        let s: any Space<BaseEnv.Observation> = env.observationSpace
        self.baseSpace = s
        guard let box = flattenSpaceToBox(s) else {
            throw GymnazoError.invalidObservationSpace
        }
        self.observationSpace = box
    }

    public mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset<MLXArray> {
        let result = try env.reset(seed: seed, options: options)
        let flattened = flatten(space: baseSpace, sample: result.obs)
        guard let flatObs = flattened as? MLXArray else {
            throw GymnazoError.invalidObservationType(
                expected: String(describing: MLXArray.self),
                actual: String(describing: type(of: flattened))
            )
        }
        return Reset(obs: flatObs, info: result.info)
    }

    public mutating func step(_ action: BaseEnv.Action) throws -> Step<MLXArray> {
        let result = try env.step(action)
        let flattened = flatten(space: baseSpace, sample: result.obs)
        guard let flatObs = flattened as? MLXArray else {
            throw GymnazoError.invalidObservationType(
                expected: String(describing: MLXArray.self),
                actual: String(describing: type(of: flattened))
            )
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
