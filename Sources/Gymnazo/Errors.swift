public enum GymnazoError: Error, Sendable, Equatable {
    case unregisteredEnvironment(id: String)
    case invalidRecordBufferLength(Int)
    case invalidMaxEpisodeSteps(Int)
    case invalidNumEnvs(Int)
    case invalidActionType(expected: String, actual: String)
    case invalidObservationType(expected: String, actual: String)
    case invalidObservationSpace
    case invalidEnvironmentType(expected: String, actual: String)
    case invalidAction(String)
    case invalidState(String)
    case invalidConfiguration(String)
    case operationFailed(String)
    case actionOutsideSpace(envId: String?)
    case observationOutsideSpace(envId: String?)
    case stepBeforeReset
    case renderBeforeReset
    case vectorEnvClosed
    case vectorEnvActionCountMismatch(expected: Int, actual: Int)
    case vectorEnvNeedsReset(index: Int)
    case vectorEnvUnsupportedActionType(actual: String)
    case vectorEnvUnsupportedObservationType(actual: String)
    case vectorEnvIncompatibleObservationShape(index: Int, expected: [Int], actual: [Int])
    case missingMaxEpisodeSteps
    case invalidFrameSkip(Int)
    case invalidStackSize(Int)
    case invalidResizeShape
    case invalidStatsKey(String)
    case invalidMap(String)
}
