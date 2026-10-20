/// A native Swift-state port of the supplied Metro Run training simulator.
///
/// Observations contain player `[32]`, entities `[64, 22]`, and a boolean entity mask `[64]`.
/// The default decision applies one command across four 60 Hz ticks. Route and gameplay
/// randomness are independent. Seeded Swift routes do not match Python's RNG sequence.
public struct MetroRunnerValueEnvironment: ValueEnvironment {
    public let valueObservationSpace: DictionaryValueSpace
    public let valueActionSpace: DiscreteValueSpace
    private var model: MetroRunnerModel

    public init(config: MetroRunnerConfig = MetroRunnerConfig()) throws {
        self.model = try MetroRunnerModel(config: config)
        self.valueObservationSpace = try MetroRunnerObservation.space()
        self.valueActionSpace = try DiscreteValueSpace(n: 6)
    }

    public mutating func resetValue(seed: UInt64?, options: ValueResetOptions?) throws -> ValueReset<[String: SpaceValue]> {
        try model.reset(seed: seed, options: options)
    }

    public mutating func stepValue(_ action: Int) throws -> ValueStep<[String: SpaceValue]> {
        try model.step(action)
    }

    public mutating func close() { model.closed = true }

    public var currentSnapshot: MetroRunnerSnapshot { model.snapshot }

    /// Adapts structured observations to the float tensor required by Gymnazo's vector trainers.
    public func flattened() throws -> TransformObservationValueEnvironment<Self, TensorValue<Float>, BoxValueSpace> {
        TransformObservationValueEnvironment(self,
            observationSpace: try BoxValueSpace(low: -1, high: 1, shape: [MetroRunnerObservation.flatCount]),
            transform: MetroRunnerObservation.flatten)
    }
}
