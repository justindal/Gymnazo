import MLX

public struct ReplayBufferAllocationInfo: Sendable, Equatable {
    public let requested: Int
    public let effective: Int

    public var wasCapped: Bool { requested != effective }
}

enum ReplayBufferSizing {
    private static let defaultMemoryBudgetBytes = 256 * 1024 * 1024

    struct Resolved: Sendable {
        let config: ReplayBuffer.Configuration
        let info: ReplayBufferAllocationInfo
    }

    static func resolve(
        config: ReplayBuffer.Configuration,
        storageObsShape: [Int],
        storageObsDtype: DType,
        actionDim: Int
    ) throws -> Resolved {
        let requested = config.bufferSize
        let observationBytes = storageObsShape.reduce(1, *) * storageObsDtype.size
        guard observationBytes > 0 else {
            return Resolved(
                config: config,
                info: ReplayBufferAllocationInfo(requested: requested, effective: requested)
            )
        }

        let transitionBytes = bytesPerTransition(
            observationBytes: observationBytes,
            actionDim: actionDim,
            optimizeMemoryUsage: config.optimizeMemoryUsage
        )
        let limit = config.memoryBudgetBytes ?? defaultMemoryBudgetBytes
        if transitionBytes > limit {
            throw GymnazoError.invalidConfiguration(
                "Replay transition requires \(transitionBytes) bytes, exceeding the memory budget of \(limit)"
            )
        }

        let maxSize = limit / transitionBytes

        if maxSize < 1 {
            throw GymnazoError.invalidConfiguration(
                "Replay buffer cannot fit on this device"
            )
        }

        let minimum = config.optimizeMemoryUsage ? 2 : 1
        if maxSize < minimum {
            throw GymnazoError.invalidConfiguration(
                "Replay buffer requires at least \(minimum) slots on this device"
            )
        }

        let effective = min(requested, maxSize)
        let info = ReplayBufferAllocationInfo(requested: requested, effective: effective)
        guard effective == requested else {
            let capped = ReplayBuffer.Configuration(
                bufferSize: effective,
                optimizeMemoryUsage: config.optimizeMemoryUsage,
                handleTimeoutTermination: config.handleTimeoutTermination,
                frameStack: config.frameStack,
                seed: config.seed,
                memoryBudgetBytes: config.memoryBudgetBytes
            )
            return Resolved(config: capped, info: info)
        }
        return Resolved(config: config, info: info)
    }

    private static func bytesPerTransition(
        observationBytes: Int,
        actionDim: Int,
        optimizeMemoryUsage: Bool
    ) -> Int {
        let observationStorage = observationBytes * (optimizeMemoryUsage ? 1 : 2)
        let actionBytes = max(1, actionDim) * MemoryLayout<Float>.size
        let scalarStorage = 3 * MemoryLayout<Float>.size
        let boundaryMetadata = MemoryLayout<Bool>.size
        return observationStorage + actionBytes + scalarStorage + boundaryMetadata
    }
}
