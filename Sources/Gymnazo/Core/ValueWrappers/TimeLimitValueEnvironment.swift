/// Truncates a typed episode after a fixed number of steps.
public struct TimeLimitValueEnvironment<Base: ValueEnvironment>: ValueEnvironment {
    public typealias Observation = Base.Observation
    public typealias Action = Base.Action
    public typealias ObservationSpace = Base.ObservationSpace
    public typealias ActionSpace = Base.ActionSpace

    public var base: Base
    public let maxEpisodeSteps: Int

    private var elapsedSteps = 0

    public var valueObservationSpace: ObservationSpace {
        base.valueObservationSpace
    }

    public var valueActionSpace: ActionSpace {
        base.valueActionSpace
    }

    /// Creates a typed time-limit wrapper.
    ///
    /// - Parameters:
    ///   - base: Environment whose episodes are limited.
    ///   - maxEpisodeSteps: Maximum number of steps per episode.
    /// - Throws: If maxEpisodeSteps is not positive.
    public init(_ base: Base, maxEpisodeSteps: Int) throws {
        guard maxEpisodeSteps > 0 else {
            throw GymnazoError.invalidMaxEpisodeSteps(maxEpisodeSteps)
        }
        self.base = base
        self.maxEpisodeSteps = maxEpisodeSteps
    }

    public mutating func resetValue(
        seed: UInt64?,
        options: ValueResetOptions?
    ) throws -> ValueReset<Observation> {
        let result = try base.resetValue(seed: seed, options: options)
        elapsedSteps = 0
        return result
    }

    public mutating func stepValue(_ action: Action) throws -> ValueStep<Observation> {
        let result = try base.stepValue(action)
        elapsedSteps += 1
        guard elapsedSteps >= maxEpisodeSteps, !result.truncated else {
            return result
        }
        var info = result.info
        info[EnvInfoKey.timeLimitTruncated] = .bool(!result.terminated)
        return ValueStep(
            observation: result.observation,
            reward: result.reward,
            terminated: result.terminated,
            truncated: true,
            info: info
        )
    }

    public mutating func close() {
        base.close()
    }
}
