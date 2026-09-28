import MLX

extension ReplayBuffer {
    mutating func addBatch(
        obs: MLXArray,
        actions actionBatch: MLXArray,
        rewards rewardBatch: MLXArray,
        nextObs: MLXArray,
        terminations: [Bool],
        truncations: [Bool]
    ) throws {
        let batchCount = terminations.count
        guard batchCount > 0 else { return }
        let observationBatch: MLXArray
        let nextObservationBatch: MLXArray
        if let frameStack = config.frameStack {
            let index = MLXArray(Int32(frameStack.size - 1))
            observationBatch = obs.take(index, axis: resolvedStackAxis + 1)
            nextObservationBatch = nextObs.take(index, axis: resolvedStackAxis + 1)
        } else {
            observationBatch = obs
            nextObservationBatch = nextObs
        }
        guard observationBatch.shape == [batchCount] + storageObsShape,
            nextObservationBatch.shape == observationBatch.shape,
            observationBatch.dtype == storageObsDtype,
            nextObservationBatch.dtype == storageObsDtype,
            actionBatch.shape == [batchCount, actionDim],
            rewardBatch.shape == [batchCount],
            truncations.count == batchCount
        else {
            throw GymnazoError.invalidConfiguration(
                "Replay batch dimensions or dtypes are invalid.")
        }
        let batchDones = zip(terminations, truncations).map { $0 || $1 }
        let doneBatch = MLXArray(batchDones.map { $0 ? Float(1) : 0 })
        let timeoutBatch = MLXArray(zip(terminations, truncations).map { !$0 && $1 ? Float(1) : 0 })
        let skipped = max(0, batchCount - bufferSize)
        if batchCount >= bufferSize - position { isBufferFull = true }
        position = (position + skipped) % bufferSize
        var start = skipped
        while start < batchCount {
            let length = min(batchCount - start, bufferSize - position)
            let source = start..<(start + length)
            let destination = position..<(position + length)
            observations[destination] = observationBatch[source]
            actions[destination] = actionBatch[source]
            rewards[destination] = rewardBatch[source]
            dones[destination] = doneBatch[source]
            timeouts[destination] = timeoutBatch[source]
            doneValues.replaceSubrange(destination, with: batchDones[source])
            if config.optimizeMemoryUsage {
                observations[(position + length) % bufferSize] =
                    nextObservationBatch[start + length - 1]
            } else {
                nextObservations![destination] = nextObservationBatch[source]
            }
            position = (position + length) % bufferSize
            start += length
        }
        if let nextObservations {
            try checkedEval(observations, nextObservations, actions, rewards, dones, timeouts)
        } else {
            try checkedEval(observations, actions, rewards, dones, timeouts)
        }
    }
}
