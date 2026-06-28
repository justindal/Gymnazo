import Foundation
import MLX

struct ReplayBufferMetadata: Codable {
    let bufferSize: Int
    let requestedBufferSize: Int?
    let optimizeMemoryUsage: Bool
    let handleTimeoutTermination: Bool
    let frameStack: ReplayBuffer.Configuration.FrameStackConfig?
    let seed: UInt64?
    let position: Int
    let isFull: Bool
    let count: Int
    let numEnvs: Int
}

extension ReplayBuffer: BufferPersisting {
    public func save(to url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)

        let n = count
        guard n > 0 else { return }

        let obsSlice = n == bufferSize ? observations : observations[0..<n]
        let actSlice = n == bufferSize ? actions : actions[0..<n]
        let rewSlice = n == bufferSize ? rewards : rewards[0..<n]
        let doneSlice = n == bufferSize ? dones : dones[0..<n]
        let timeoutSlice = n == bufferSize ? timeouts : timeouts[0..<n]

        try MLXEval.evaluate(obsSlice, actSlice, rewSlice, doneSlice, timeoutSlice)

        try MLX.save(
            arrays: ["data": obsSlice],
            url: url.appendingPathComponent("observations.safetensors"))
        try MLX.save(
            arrays: ["data": actSlice],
            url: url.appendingPathComponent("actions.safetensors"))
        try MLX.save(
            arrays: ["data": rewSlice],
            url: url.appendingPathComponent("rewards.safetensors"))
        try MLX.save(
            arrays: ["data": doneSlice],
            url: url.appendingPathComponent("dones.safetensors"))
        try MLX.save(
            arrays: ["data": timeoutSlice],
            url: url.appendingPathComponent("timeouts.safetensors"))

        if let nextObs = nextObservations {
            let nextSlice = n == bufferSize ? nextObs : nextObs[0..<n]
            try MLXEval.evaluate(nextSlice)
            try MLX.save(
                arrays: ["data": nextSlice],
                url: url.appendingPathComponent("next_observations.safetensors"))
        }

        let meta = ReplayBufferMetadata(
            bufferSize: config.bufferSize,
            requestedBufferSize: allocationInfo.requested,
            optimizeMemoryUsage: config.optimizeMemoryUsage,
            handleTimeoutTermination: config.handleTimeoutTermination,
            frameStack: config.frameStack,
            seed: config.seed,
            position: position,
            isFull: isBufferFull,
            count: n,
            numEnvs: numEnvs
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let metaData = try encoder.encode(meta)
        try metaData.write(to: url.appendingPathComponent("buffer_meta.json"))
    }

    public mutating func load(from url: URL) throws {
        let metaURL = url.appendingPathComponent("buffer_meta.json")
        guard FileManager.default.fileExists(atPath: metaURL.path) else {
            throw PersistenceError.missingFile("buffer_meta.json")
        }

        let metaData = try Data(contentsOf: metaURL)
        let meta = try JSONDecoder().decode(ReplayBufferMetadata.self, from: metaData)

        try validateMetadata(meta)

        guard meta.count > 0 else {
            reset()
            return
        }

        let n = bufferSize

        observations = try Self.loadAndPad("observations.safetensors", from: url, to: n)
        actions = try Self.loadAndPad("actions.safetensors", from: url, to: n)
        rewards = try Self.loadAndPad("rewards.safetensors", from: url, to: n)
        dones = try Self.loadAndPad("dones.safetensors", from: url, to: n)
        timeouts = try Self.loadAndPad("timeouts.safetensors", from: url, to: n)

        if !config.optimizeMemoryUsage {
            let nextURL = url.appendingPathComponent("next_observations.safetensors")
            if FileManager.default.fileExists(atPath: nextURL.path) {
                nextObservations = try Self.loadAndPad(
                    "next_observations.safetensors", from: url, to: n)
            }
        }

        try validateLoadedTensors()

        var toEval = [observations, actions, rewards, dones, timeouts]
        if let next = nextObservations { toEval.append(next) }
        try MLXEval.evaluate(toEval)

        position = meta.position
        isBufferFull = meta.isFull
    }

    private func validateMetadata(_ meta: ReplayBufferMetadata) throws {
        guard meta.bufferSize == config.bufferSize else {
            throw PersistenceError.invalidCheckpoint("bufferSize mismatch")
        }
        guard meta.optimizeMemoryUsage == config.optimizeMemoryUsage else {
            throw PersistenceError.invalidCheckpoint("optimizeMemoryUsage mismatch")
        }
        guard meta.handleTimeoutTermination == config.handleTimeoutTermination else {
            throw PersistenceError.invalidCheckpoint("handleTimeoutTermination mismatch")
        }
        guard meta.frameStack == config.frameStack else {
            throw PersistenceError.invalidCheckpoint("frameStack mismatch")
        }
    }

    private func validateLoadedTensors() throws {
        guard observations.shape[0] == bufferSize else {
            throw PersistenceError.invalidCheckpoint("observations buffer dimension mismatch")
        }
        let expectedObsShape = [bufferSize] + storageObsShape
        guard observations.shape == expectedObsShape else {
            throw PersistenceError.invalidCheckpoint("observations shape mismatch")
        }
        guard actions.shape[0] == bufferSize else {
            throw PersistenceError.invalidCheckpoint("actions buffer dimension mismatch")
        }
        if let nextObs = nextObservations {
            guard nextObs.shape == expectedObsShape else {
                throw PersistenceError.invalidCheckpoint("next_observations shape mismatch")
            }
        }
    }

    private static func loadAndPad(
        _ name: String, from url: URL, to bufferSize: Int
    ) throws -> MLXArray {
        let fileURL = url.appendingPathComponent(name)
        guard let data = try MLX.loadArrays(url: fileURL)["data"] else {
            throw PersistenceError.invalidCheckpoint("Missing \(name)")
        }
        if data.shape[0] >= bufferSize {
            return data[0..<bufferSize]
        }
        let padShape = [bufferSize - data.shape[0]] + Array(data.shape.dropFirst())
        return MLX.concatenated([data, MLX.zeros(padShape, dtype: data.dtype)], axis: 0)
    }
}
