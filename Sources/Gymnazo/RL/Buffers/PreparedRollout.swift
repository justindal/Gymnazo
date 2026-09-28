import MLX

struct PreparedRollout {
    let batch: RolloutBatch

    func batches(batchSize: Int, generator: inout SeededGenerator) -> Batches {
        let sampleCount = batch.values.size
        var permutation = Array(0..<sampleCount)
        if sampleCount > 1 {
            for index in stride(from: sampleCount - 1, through: 1, by: -1) {
                let next = Int(generator.next() % UInt64(index + 1))
                permutation.swapAt(index, next)
            }
        }
        return Batches(
            batch: batch,
            permutation: permutation.map(Int32.init),
            batchSize: max(1, min(batchSize, sampleCount))
        )
    }

    struct Batches: Sequence, IteratorProtocol {
        let batch: RolloutBatch
        let permutation: [Int32]
        let batchSize: Int
        private var start = 0

        init(batch: RolloutBatch, permutation: [Int32], batchSize: Int) {
            self.batch = batch
            self.permutation = permutation
            self.batchSize = batchSize
        }

        mutating func next() -> RolloutBatch? {
            guard start < permutation.count else { return nil }
            let end = Swift.min(start + batchSize, permutation.count)
            let indices = MLXArray(Array(permutation[start..<end]))
            start = end
            return RolloutBatch(
                observations: batch.observations[indices],
                actions: batch.actions[indices],
                values: batch.values[indices],
                logProbs: batch.logProbs[indices],
                advantages: batch.advantages[indices],
                returns: batch.returns[indices]
            )
        }
    }
}
