import MLX

public protocol Batchable {
    static func stack(_ items: [Self]) -> Self
}

extension MLXArray: Batchable {
    public static func stack(_ items: [MLXArray]) -> MLXArray {
        MLX.stacked(items)
    }
}

extension Dictionary: Batchable where Key == String, Value == MLXArray {
    public static func stack(_ items: [Self]) -> Self {
        guard let first: [String: MLXArray] = items.first else { return [:] }
        var result: [String: MLXArray] = [:]
        for key: String in first.keys {
            result[key] = MLX.stacked(items.compactMap { $0[key] })
        }
        return result
    }
}

public struct Transition<Obs> {
    public let obs: Obs
    public let action: MLXArray
    public let reward: MLXArray
    public let nextObs: Obs
    public let done: MLXArray
    public let truncated: MLXArray

    public init(
        obs: Obs,
        action: MLXArray,
        reward: MLXArray,
        nextObs: Obs,
        done: MLXArray,
        truncated: MLXArray
    ) {
        self.obs = obs
        self.action = action
        self.reward = reward
        self.nextObs = nextObs
        self.done = done
        self.truncated = truncated
    }
}

public struct Sample<Obs> {
    public let obs: Obs
    public let actions: MLXArray
    public let rewards: MLXArray
    public let nextObs: Obs
    public let dones: MLXArray
    public let truncateds: MLXArray
}

public struct ReplayBuffer<Obs: Batchable> {
    public let capacity: Int
    private var storage: CircularBuffer<Transition<Obs>>

    public var count: Int { storage.count }
    public var isEmpty: Bool { storage.isEmpty }
    public var isFull: Bool { storage.isFull }

    public init(capacity: Int) {
        self.capacity = capacity
        self.storage = CircularBuffer(capacity: capacity)
    }

    public mutating func add(_ transition: Transition<Obs>) {
        storage.add(transition)
    }

    public mutating func reset() {
        storage.reset()
    }

    public func sample(_ batchSize: Int) -> Sample<Obs> {
        precondition(count >= batchSize, "Not enough samples")

        let batch: [Transition<Obs>] = storage.sample(batchSize)

        return Sample(
            obs: Obs.stack(batch.map(\.obs)),
            actions: MLXArray.stack(batch.map(\.action)),
            rewards: MLXArray.stack(batch.map(\.reward)),
            nextObs: Obs.stack(batch.map(\.nextObs)),
            dones: MLXArray.stack(batch.map(\.done)),
            truncateds: MLXArray.stack(batch.map(\.truncated))
        )
    }
}

public struct CircularBuffer<T> {
    private var data: [T?]
    private var head: Int = 0
    private var filled: Bool = false

    public let capacity: Int
    public var count: Int { filled ? capacity : head }
    public var isEmpty: Bool { count == 0 }
    public var isFull: Bool { filled }

    public init(capacity: Int) {
        self.capacity = capacity
        self.data = Array(repeating: nil, count: capacity)
    }

    public mutating func add(_ element: T) {
        data[head] = element
        head = (head + 1) % capacity
        if head == 0 { filled = true }
    }

    public mutating func reset() {
        data = Array(repeating: nil, count: capacity)
        head = 0
        filled = false
    }

    public subscript(i: Int) -> T {
        data[i]!
    }

    public func sample(_ n: Int) -> [T] {
        (0..<n).map { _ in self[Int.random(in: 0..<count)] }
    }
}
