import MLX
import MLXNN

enum MLXEval {
    static func evaluate(_ array: MLXArray) throws {
        try checkedEval(array)
    }

    static func evaluate(_ arrays: [MLXArray]) throws {
        guard !arrays.isEmpty else { return }
        try checkedEval(arrays)
    }

    static func evaluate(_ first: MLXArray, _ rest: MLXArray...) throws {
        if rest.isEmpty {
            try evaluate(first)
        } else {
            try checkedEval([first] + rest)
        }
    }
}
