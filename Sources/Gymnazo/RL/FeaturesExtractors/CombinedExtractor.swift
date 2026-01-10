//
//  CombinedExtractor.swift
//  Gymnazo
//
//

import MLX
import MLXNN

/// Combined features extractor for Dict observation spaces.
/// Builds a features extractor for each key of the space.
public final class CombinedExtractor: Module, DictFeaturesExtractor {
    public let featuresDim: Int

    private let keys: [String]
    private let normalizedImage: Bool
    private let cnnOutputDim: Int

    @ModuleInfo private var extractors: [String: any UnaryLayer]

    public init(
        observationSpace: Dict,
        featuresDim: Int,
        normalizedImage: Bool = false,
        cnnOutputDim: Int = 256
    ) {
        precondition(cnnOutputDim > 0)
        self.cnnOutputDim = cnnOutputDim
        self.normalizedImage = normalizedImage

        let sortedKeys = observationSpace.spaces.keys.sorted()
        self.keys = sortedKeys

        var ex: [String: any UnaryLayer] = [:]
        var total = 0

        for k in sortedKeys {
            guard let subspace = observationSpace.spaces[k] else { continue }

            if let box = subspace as? Box,
                CombinedExtractor.isImageSpace(
                    box,
                    normalizedImage: normalizedImage
                )
            {
                ex[k] = NatureCNN(
                    observationSpace: box,
                    featuresDim: cnnOutputDim,
                    normalizedImage: normalizedImage
                )
                total += cnnOutputDim
            } else if let box = subspace as? Box {
                ex[k] = FlattenToBatch()
                total += CombinedExtractor.flattenedObsDim(box)
            } else {
                preconditionFailure("Unsupported")
            }
        }

        self.extractors = ex
        self.featuresDim = total

        super.init()
    }

    public func callAsFunction(_ observations: [String: MLXArray]) -> MLXArray {
        var encoded: [MLXArray] = []
        encoded.reserveCapacity(keys.count)

        for k in keys {
            guard let x = observations[k] else {
                preconditionFailure(
                    "CombinedExtractor: missing observation for key '\(k)'"
                )
            }
            guard let extractor = extractors[k] else {
                preconditionFailure(
                    "CombinedExtractor: missing extractor for key '\(k)'"
                )
            }

            var y = extractor(x)

            // Ensure [B, F] so concatenation along axis 1 is valid.
            if y.shape.count == 1 {
                y = y.reshaped([1, y.shape[0]])
            } else if y.shape.count > 2 {
                let b = y.shape[0]
                let f = y.shape.dropFirst().reduce(1, *)
                y = y.reshaped([b, f])
            }

            encoded.append(y)
        }

        return concatenated(encoded, axis: 1)
    }

    /// Must be Box with 3 dims.
    /// If not normalized, expect uint8.
    private static func isImageSpace(_ box: Box, normalizedImage: Bool) -> Bool
    {
        guard let shp = box.shape, shp.count == 3 else { return false }
        if normalizedImage { return true }
        return box.dtype == .uint8
    }

    /// Flattens inputs into `[B, F]`:
    /// - `[F]` -> `[1, F]`
    /// - `[B, ...]` -> `[B, F]`
    private class FlattenToBatch: Module, UnaryLayer {
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            if x.shape.count == 1 { return x.reshaped([1, x.shape[0]]) }
            if x.shape.count == 2 { return x }
            let b = x.shape[0]
            let f = x.shape.dropFirst().reduce(1, *)
            return x.reshaped([b, f])
        }
    }

    /// Flatten
    private static func flattenedObsDim(_ box: Box) -> Int {
        let shp = box.shape ?? box.low.shape
        return shp.reduce(1, *)
    }
}
