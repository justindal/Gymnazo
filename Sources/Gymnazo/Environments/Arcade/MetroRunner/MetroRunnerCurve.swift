import Foundation

struct MetroRunnerCurve: Decodable, Sendable, Equatable {
    let keys: [Key]

    struct Key: Decodable, Sendable, Equatable {
        let time: Double
        let value: Double
        let inSlope: Double
        let outSlope: Double
    }

    func value(at time: Double) -> Double {
        guard let first = keys.first, let last = keys.last else { return 0 }
        if time <= first.time { return first.value }
        if time >= last.time { return last.value }
        let index = keys.firstIndex { $0.time > time } ?? keys.count - 1
        let a = keys[index - 1], b = keys[index]
        if a.outSlope.isInfinite || b.inSlope.isInfinite { return a.value }
        let span = b.time - a.time
        let t = (time - a.time) / span, t2 = t * t, t3 = t2 * t
        return (2 * t3 - 3 * t2 + 1) * a.value + (t3 - 2 * t2 + t) * span * a.outSlope
            + (-2 * t3 + 3 * t2) * b.value + (t3 - t2) * span * b.inSlope
    }
}
