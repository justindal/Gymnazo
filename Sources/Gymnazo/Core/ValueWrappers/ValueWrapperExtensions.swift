extension ValueEnvironment {
    /// Validates reset, step, and close lifecycle transitions.
    ///
    /// - Returns: A lifecycle-checking wrapper.
    public func checked() -> CheckedValueEnvironment<Self> {
        CheckedValueEnvironment(self)
    }

    /// Limits each episode to a maximum number of steps.
    ///
    /// - Parameter maxEpisodeSteps: Maximum number of steps per episode.
    /// - Returns: A time-limit wrapper.
    /// - Throws: If maxEpisodeSteps is not positive.
    public func timeLimited(_ maxEpisodeSteps: Int) throws -> TimeLimitValueEnvironment<Self> {
        try TimeLimitValueEnvironment(self, maxEpisodeSteps: maxEpisodeSteps)
    }

    /// Applies a transform to each reward.
    ///
    /// - Parameter transform: Closure that maps each reward to its replacement.
    /// - Returns: A reward-transforming wrapper.
    public func rewardsTransformed(
        _ transform: @escaping @Sendable (Double) throws -> Double
    ) -> TransformRewardValueEnvironment<Self> {
        TransformRewardValueEnvironment(self, transform: transform)
    }

    /// Clips rewards to optional lower and upper bounds.
    ///
    /// - Parameters:
    ///   - minimum: Optional lower reward bound.
    ///   - maximum: Optional upper reward bound.
    /// - Returns: A reward-clipping wrapper.
    /// - Throws: If both bounds are supplied and minimum exceeds maximum.
    public func rewardsClipped(
        min minimum: Double? = nil,
        max maximum: Double? = nil
    ) throws -> ClipRewardValueEnvironment<Self> {
        if let minimum, let maximum, minimum > maximum {
            throw GymnazoError.invalidConfiguration("Reward clipping bounds are reversed.")
        }
        return TransformRewardValueEnvironment(self) { reward in
            Swift.min(Swift.max(reward, minimum ?? -.infinity), maximum ?? .infinity)
        }
    }

    /// Transforms observations and replaces their value space.
    ///
    /// - Parameters:
    ///   - observationSpace: Space for transformed observations.
    ///   - transform: Closure that maps source observations to output values.
    /// - Returns: An observation-transforming wrapper.
    public func observationsTransformed<Output: Sendable, OutputSpace: ValueSpace>(
        to observationSpace: OutputSpace,
        _ transform: @escaping @Sendable (Observation) throws -> Output
    ) -> TransformObservationValueEnvironment<Self, Output, OutputSpace>
    where OutputSpace.Value == Output {
        TransformObservationValueEnvironment(
            self,
            observationSpace: observationSpace,
            transform: transform
        )
    }

    /// Transforms observations using a typed output space.
    ///
    /// - Parameters:
    ///   - observationSpace: Space for transformed observations.
    ///   - transform: Closure that maps source observations to output values.
    /// - Returns: A typed observation wrapper.
    public func observationsCasted<Output: Sendable, OutputSpace: ValueSpace>(
        to observationSpace: OutputSpace,
        _ transform: @escaping @Sendable (Observation) throws -> Output
    ) -> DtypeObservationValueEnvironment<Self, Output, OutputSpace>
    where OutputSpace.Value == Output {
        observationsTransformed(to: observationSpace, transform)
    }

    /// Transforms typed actions before passing them to the wrapped environment.
    ///
    /// - Parameters:
    ///   - actionSpace: Space for actions accepted by the wrapper.
    ///   - transform: Closure that maps wrapper actions to base actions.
    /// - Returns: An action-transforming wrapper.
    public func actionsTransformed<Input: Sendable, InputSpace: ValueSpace>(
        from actionSpace: InputSpace,
        _ transform: @escaping @Sendable (Input) throws -> Action
    ) -> TransformActionValueEnvironment<Self, Input, InputSpace>
    where InputSpace.Value == Input {
        TransformActionValueEnvironment(
            self,
            actionSpace: actionSpace,
            transform: transform
        )
    }

    /// Records episode returns, lengths, and rolling statistics.
    ///
    /// - Parameters:
    ///   - bufferLength: Maximum number of completed episodes retained.
    ///   - statsKey: Info key used for the statistics dictionary.
    /// - Returns: An episode-statistics wrapper.
    /// - Throws: If bufferLength is not positive.
    public func recordingStatistics(
        bufferLength: Int = 100,
        statsKey: String = EnvInfoKey.episode
    ) throws -> RecordEpisodeStatisticsValueEnvironment<Self> {
        try RecordEpisodeStatisticsValueEnvironment(
            self,
            bufferLength: bufferLength,
            statsKey: statsKey
        )
    }
}

extension ValueEnvironment
where Observation == TensorValue<Float>, ObservationSpace == BoxValueSpace {
    /// Reshapes tensor observations to the supplied shape.
    ///
    /// - Parameter shape: Target observation shape.
    /// - Returns: A reshape wrapper.
    /// - Throws: If shape dimensions are invalid or change the element count.
    public func observationsReshaped(
        to shape: [Int]
    ) throws -> ReshapeObservationValueEnvironment<Self> {
        try ReshapeObservationValueEnvironment(self, shape: shape)
    }

    /// Delays tensor observations by a fixed number of steps.
    ///
    /// - Parameter delay: Number of prior observations to retain.
    /// - Returns: An observation-delay wrapper.
    /// - Throws: If delay is negative.
    public func observationsDelayed(
        by delay: Int
    ) throws -> DelayObservationValueEnvironment<Self> {
        try DelayObservationValueEnvironment(self, delay: delay)
    }

    /// Appends normalized or raw elapsed time to observations.
    ///
    /// - Parameters:
    ///   - maxEpisodeSteps: Episode length used for time normalization.
    ///   - normalizeTime: Whether appended time is normalized to zero through one.
    /// - Returns: A time-aware observation wrapper.
    /// - Throws: If maxEpisodeSteps is not positive.
    public func timeAware(
        maxEpisodeSteps: Int,
        normalizeTime: Bool = false
    ) throws -> TimeAwareObservationValueEnvironment<Self> {
        try TimeAwareObservationValueEnvironment(
            self,
            maxEpisodeSteps: maxEpisodeSteps,
            normalizeTime: normalizeTime
        )
    }

    /// Max-pools observations over skipped steps.
    ///
    /// - Parameter skip: Number of repeated actions per wrapper step.
    /// - Returns: A max-and-skip wrapper.
    /// - Throws: If skip is not positive.
    public func maxAndSkip(
        _ skip: Int = 4
    ) throws -> MaxAndSkipObservationValueEnvironment<Self> {
        try MaxAndSkipObservationValueEnvironment(self, skip: skip)
    }
}

extension ValueEnvironment where Action: Equatable {
    /// Repeats the previous action with the supplied probability.
    ///
    /// - Parameters:
    ///   - probability: Probability of repeating the previous action.
    ///   - seed: Seed for the repeat decision generator.
    /// - Returns: A sticky-action wrapper.
    /// - Throws: If probability is outside zero through one.
    public func stickyActions(
        probability: Float,
        seed: UInt64 = 0
    ) throws -> StickyActionValueEnvironment<Self> {
        try StickyActionValueEnvironment(
            self,
            repeatProbability: probability,
            seed: seed
        )
    }
}

extension ValueEnvironment
where Observation == [String: SpaceValue], ObservationSpace == DictionaryValueSpace {
    /// Keeps only the selected keys in dictionary observations.
    ///
    /// - Parameter keys: Dictionary keys to retain.
    /// - Returns: A dictionary observation filter.
    /// - Throws: If keys are duplicated or unknown.
    public func filteringObservation(
        keys: [String]
    ) throws -> FilterDictionaryObservationValueEnvironment<Self> {
        try FilterDictionaryObservationValueEnvironment(self, keys: keys)
    }
}

extension ValueEnvironment
where Observation == [SpaceValue], ObservationSpace == TupleValueSpace {
    /// Keeps only the selected indices in tuple observations.
    ///
    /// - Parameter indices: Tuple indices to retain.
    /// - Returns: A tuple observation filter.
    /// - Throws: If indices are duplicated or outside the tuple space.
    public func filteringObservation(
        indices: [Int]
    ) throws -> FilterTupleObservationValueEnvironment<Self> {
        try FilterTupleObservationValueEnvironment(self, indices: indices)
    }
}

extension ValueEnvironment where ObservationSpace: SpaceValueConvertible {
    /// Resets automatically after a terminated or truncated step.
    ///
    /// - Parameter mode: When completed episodes are reset.
    /// - Returns: An autoreset wrapper.
    public func autoReset(
        mode: AutoresetMode = .nextStep
    ) -> AutoResetValueEnvironment<Self> {
        AutoResetValueEnvironment(self, mode: mode)
    }
}
