//
//  LogEntropyCoefModule.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Module wrapper for trainable log entropy coefficient.
///
/// This lets us use an optimizer.
final class LogEntropyCoefModule: Module, @unchecked Sendable {
    @ModuleInfo var logEntCoef: MLXArray

    init(initialValue: Float) {
        self.logEntCoef = MLX.log(MLXArray([initialValue]))
        super.init()
    }

    func callAsFunction() -> MLXArray {
        logEntCoef
    }

    var value: MLXArray { logEntCoef }
    var entCoef: MLXArray { MLX.exp(logEntCoef) }
}
