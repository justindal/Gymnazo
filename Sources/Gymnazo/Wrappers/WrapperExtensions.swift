//
//  WrapperExtensions.swift
//

import MLX

/// Chainable wrapper methods for environments.
///
/// These extensions allow method chaining for a cleaner wrapper syntax:
///
/// ```swift
/// let env = CartPole()
///     .orderEnforced()
///     .recordingStatistics()
///     .timeLimited(500)
/// ```
///
/// Instead of the nested constructor style:
///
/// ```swift
/// let env = TimeLimit(env: RecordEpisodeStatistics(env: OrderEnforcing(env: CartPole())))
/// ```

public extension Env {
    
    /// Wraps the environment with order enforcement.
    ///
    /// Ensures `reset()` is called before `step()` or `render()`.
    ///
    /// - Parameter disableRenderOrderEnforcing: If `true`, allows `render()` before `reset()`.
    /// - Returns: The wrapped environment.
    func orderEnforced(
        disableRenderOrderEnforcing: Bool = false
    ) -> OrderEnforcing<Self> {
        OrderEnforcing(env: self, disableRenderOrderEnforcing: disableRenderOrderEnforcing)
    }
    
    /// Wraps the environment with passive environment checking.
    ///
    /// Runs validation on the first `reset()`/`step()`/`render()` call.
    ///
    /// - Returns: The wrapped environment.
    func passiveChecked() -> PassiveEnvChecker<Self> {
        PassiveEnvChecker(env: self)
    }
    
    /// Wraps the environment with episode statistics recording.
    ///
    /// Tracks cumulative rewards, episode lengths, and elapsed time.
    /// Attaches an "episode" entry to the `info` dictionary on termination.
    ///
    /// - Parameters:
    ///   - bufferLength: Maximum number of episodes to keep in the statistics buffer.
    ///   - statsKey: The key used in the info dictionary for episode stats.
    /// - Returns: The wrapped environment.
    func recordingStatistics(
        bufferLength: Int = 100,
        statsKey: String = "episode"
    ) -> RecordEpisodeStatistics<Self> {
        RecordEpisodeStatistics(env: self, bufferLength: bufferLength, statsKey: statsKey)
    }
    
    /// Wraps the environment with a time limit.
    ///
    /// Enforces a maximum number of steps per episode by emitting
    /// the truncation signal once the configured limit is reached.
    ///
    /// - Parameter maxSteps: Maximum steps before truncation.
    /// - Returns: The wrapped environment.
    func timeLimited(_ maxSteps: Int) -> TimeLimit<Self> {
        TimeLimit(env: self, maxEpisodeSteps: maxSteps)
    }

    func rewardsTransformed(
        _ transform: @escaping (Double) -> Double
    ) -> TransformReward<Self> {
        TransformReward(env: self, transform: transform)
    }

    func rewardsNormalized(
        gamma: Double = 0.99,
        epsilon: Double = 1e-8
    ) -> NormalizeReward<Self> {
        NormalizeReward(env: self, gamma: gamma, epsilon: epsilon)
    }

    func autoReset(
        mode: AutoresetMode = .nextStep
    ) -> AutoReset<Self> {
        AutoReset(env: self, mode: mode)
    }

    func observationsFlattened() -> FlattenObservation<Self> {
        FlattenObservation(env: self)
    }
}

public extension Env where Observation == MLXArray {
    
    /// Wraps the environment with observation normalization.
    ///
    /// Normalizes observations to approximately mean 0 and variance 1
    /// using a running mean and variance estimator.
    ///
    /// - Returns: The wrapped environment.
    func observationsNormalized() -> NormalizeObservation<Self> {
        NormalizeObservation(env: self)
    }
    
    /// Wraps the environment with a custom observation transform.
    ///
    /// - Parameters:
    ///   - observationSpace: Optional new observation space. Defaults to the original.
    ///   - transform: A closure that transforms each observation.
    /// - Returns: The wrapped environment.
    func observationsTransformed(
        observationSpace: ObservationSpace? = nil,
        _ transform: @escaping (MLXArray) -> MLXArray
    ) -> TransformObservation<Self> {
        TransformObservation(env: self, transform: transform, observation_space: observationSpace)
    }
}

public extension Env where ActionSpace == Box {
    
    /// Wraps the environment with action clipping.
    ///
    /// Clips actions to the Box action space bounds before passing to the environment.
    ///
    /// - Returns: The wrapped environment.
    func actionsClipped() -> ClipAction<Self> {
        ClipAction(env: self)
    }
    
    /// Wraps the environment with action rescaling.
    ///
    /// Rescales actions from a source range into the Box action space bounds.
    ///
    /// - Parameters:
    ///   - sourceRange: The source range. Default is `(-1.0, 1.0)`.
    /// - Returns: The wrapped environment.
    func actionsRescaled(
        from sourceRange: (low: Float, high: Float) = (-1.0, 1.0)
    ) -> RescaleAction<Self> {
        RescaleAction(env: self, sourceLow: sourceRange.low, sourceHigh: sourceRange.high)
    }
}

public extension Env {
    
    /// Applies multiple wrappers in sequence using a closure.
    ///
    /// This is useful when you need conditional wrapping or complex logic:
    ///
    /// ```swift
    /// let env = CartPole().wrapped { env in
    ///     var wrapped = env.orderEnforced()
    ///     if enableStats {
    ///         return wrapped.recordingStatistics()
    ///     }
    ///     return wrapped
    /// }
    /// ```
    ///
    /// - Parameter transform: A closure that takes this environment and returns a wrapped version.
    /// - Returns: The result of the transform closure.
    func wrapped<Wrapped: Env>(_ transform: (Self) -> Wrapped) -> Wrapped {
        transform(self)
    }
}

public extension Env {
    
    /// Applies the standard validation wrappers.
    ///
    /// This applies (in order):
    /// 1. `PassiveEnvChecker` - validates environment conformance
    /// 2. `OrderEnforcing` - ensures reset() before step()
    ///
    /// - Returns: The wrapped environment.
    func validated() -> OrderEnforcing<PassiveEnvChecker<Self>> {
        self.passiveChecked().orderEnforced()
    }
    
    /// Applies the standard validation wrappers with a time limit.
    ///
    /// This applies (in order):
    /// 1. `PassiveEnvChecker` - validates environment conformance
    /// 2. `OrderEnforcing` - ensures reset() before step()
    /// 3. `TimeLimit` - enforces max episode steps
    ///
    /// - Parameter maxSteps: Maximum steps per episode.
    /// - Returns: The wrapped environment.
    func validated(
        maxSteps: Int
    ) -> TimeLimit<OrderEnforcing<PassiveEnvChecker<Self>>> {
        self.passiveChecked()
            .orderEnforced()
            .timeLimited(maxSteps)
    }
}
