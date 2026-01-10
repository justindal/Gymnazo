//
//  PolyakUpdate.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Update target parameters using Polyak.
///
/// - Parameters:
///   - target: Current target parameters.
///   - source: Source parameters.
///   - tau: Interpolation factor in `[0, 1]`.
/// - Returns: Updated target parameters.
func polyakUpdate(
    target: ModuleParameters,
    source: ModuleParameters,
    tau: Double
) -> ModuleParameters {
    let tauF = Float(tau)
    let targetFlat = Dictionary(uniqueKeysWithValues: target.flattened())
    let sourceFlat = Dictionary(uniqueKeysWithValues: source.flattened())

    var updated: [String: MLXArray] = [:]
    for (key, tArr) in targetFlat {
        if let sArr = sourceFlat[key] {
            updated[key] = (1.0 - tauF) * tArr + tauF * sArr
        } else {
            updated[key] = tArr
        }
    }

    return ModuleParameters.unflattened(updated)
}
