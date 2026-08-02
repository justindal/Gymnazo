extension ValueEnvironment {
    public func checked() -> CheckedValueEnvironment<Self> {
        CheckedValueEnvironment(self)
    }

    public func timeLimited(_ maxEpisodeSteps: Int) throws -> TimeLimitValueEnvironment<Self> {
        try TimeLimitValueEnvironment(self, maxEpisodeSteps: maxEpisodeSteps)
    }

    public func rewardsTransformed(
        _ transform: @escaping @Sendable (Double) throws -> Double
    ) -> TransformRewardValueEnvironment<Self> {
        TransformRewardValueEnvironment(self, transform: transform)
    }

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

    public func observationsCasted<Output: Sendable, OutputSpace: ValueSpace>(
        to observationSpace: OutputSpace,
        _ transform: @escaping @Sendable (Observation) throws -> Output
    ) -> DtypeObservationValueEnvironment<Self, Output, OutputSpace>
    where OutputSpace.Value == Output {
        observationsTransformed(to: observationSpace, transform)
    }

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
    public func observationsReshaped(
        to shape: [Int]
    ) throws -> ReshapeObservationValueEnvironment<Self> {
        try ReshapeObservationValueEnvironment(self, shape: shape)
    }

    public func observationsDelayed(
        by delay: Int
    ) throws -> DelayObservationValueEnvironment<Self> {
        try DelayObservationValueEnvironment(self, delay: delay)
    }

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

    public func maxAndSkip(
        _ skip: Int = 4
    ) throws -> MaxAndSkipObservationValueEnvironment<Self> {
        try MaxAndSkipObservationValueEnvironment(self, skip: skip)
    }
}

extension ValueEnvironment where Action: Equatable {
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
    public func filteringObservation(
        keys: [String]
    ) throws -> FilterDictionaryObservationValueEnvironment<Self> {
        try FilterDictionaryObservationValueEnvironment(self, keys: keys)
    }
}

extension ValueEnvironment
where Observation == [SpaceValue], ObservationSpace == TupleValueSpace {
    public func filteringObservation(
        indices: [Int]
    ) throws -> FilterTupleObservationValueEnvironment<Self> {
        try FilterTupleObservationValueEnvironment(self, indices: indices)
    }
}

extension ValueEnvironment where ObservationSpace: SpaceValueConvertible {
    public func autoReset(
        mode: AutoresetMode = .nextStep
    ) -> AutoResetValueEnvironment<Self> {
        AutoResetValueEnvironment(self, mode: mode)
    }
}
