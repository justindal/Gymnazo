import MLX
import MLXNN

enum MLXEval {
    static func run<R>(_ body: () throws -> R) throws -> R {
        try withError { try body() }
    }

    static func evaluate(_ array: MLXArray) throws {
        try run { eval(array) }
    }

    static func evaluate(_ arrays: [MLXArray]) throws {
        guard !arrays.isEmpty else { return }
        try run { eval(arrays) }
    }

    static func evaluate(_ first: MLXArray, _ rest: MLXArray...) throws {
        if rest.isEmpty {
            try evaluate(first)
        } else {
            try run { eval([first] + rest) }
        }
    }
}
