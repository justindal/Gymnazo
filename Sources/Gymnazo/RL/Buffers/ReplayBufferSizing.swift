import MLX

public struct ReplayBufferAllocationInfo: Sendable, Equatable {
    public let requested: Int
    public let effective: Int

    public var wasCapped: Bool { requested != effective }
}

enum ReplayBufferSizing {
    private static let fallbackMaxBytes = 256 * 1024 * 1024

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
        let bytesPerObs = storageObsShape.reduce(1, *) * storageObsDtype.size
        guard bytesPerObs > 0 else {
            return Resolved(
                config: config,
                info: ReplayBufferAllocationInfo(requested: requested, effective: requested)
            )
        }

        let limit = deviceMaxBytes()
        if bytesPerObs > limit {
            throw GymnazoError.invalidConfiguration(
                "Observation frame requires \(bytesPerObs) bytes, exceeding the device limit of \(limit)"
            )
        }

        let actionBytes = max(1, actionDim) * MemoryLayout<Float>.size
        let scalarBytes = MemoryLayout<Float>.size
        var maxSize = limit / bytesPerObs
        maxSize = min(maxSize, limit / actionBytes)
        maxSize = min(maxSize, limit / scalarBytes)

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
                seed: config.seed
            )
            return Resolved(config: capped, info: info)
        }
        return Resolved(config: config, info: info)
    }

    private static func deviceMaxBytes() -> Int {
        let max = GPU.deviceInfo().maxBufferSize
        return max > 0 ? max : fallbackMaxBytes
    }
}
