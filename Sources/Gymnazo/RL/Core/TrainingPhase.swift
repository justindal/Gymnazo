/// The current stage of a training operation, without an estimated completion percentage.
public enum TrainingPhase: String, Sendable {
    case preparing
    case collecting
    case compiling
    case updating
}
