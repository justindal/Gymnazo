//
//  Tuple.swift
//

import Foundation
import MLX

private struct AnySpaceBox {
    let base: any Space
    let shape: [Int]?
    let dtype: DType?
    let sample: (MLXArray, MLXArray?, MLXArray?) -> Any
    let contains: (Any) -> Bool

    init<S: Space>(_ space: S) {
        self.base = space
        self.shape = space.shape
        self.dtype = space.dtype
        self.sample = { key, mask, probability in
            space.sample(key: key, mask: mask, probability: probability)
        }
        self.contains = { value in
            guard let castValue = value as? S.T else {
                return false
            }
            return space.contains(castValue)
        }
    }
}

private func makeBox(_ space: any Space) -> AnySpaceBox {
    func build<S: Space>(_ space: S) -> AnySpaceBox {
        AnySpaceBox(space)
    }
    return build(space)
}

public struct Tuple: Space {
    public typealias T = [Any]

    public let spaces: [any Space]
    private let boxes: [AnySpaceBox]

    public init(_ spaces: [any Space]) {
        self.spaces = spaces
        self.boxes = spaces.map { makeBox($0) }
    }

    public init(_ spaces: any Space...) {
        self.spaces = spaces
        self.boxes = spaces.map { makeBox($0) }
    }

    public var shape: [Int]? {
        return nil
    }

    public var dtype: DType? {
        return nil
    }

    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> [Any] {
        let keys = MLX.split(key: key, into: boxes.count)
        var sample: [Any] = []

        for (i, space) in boxes.enumerated() {
            sample.append(space.sample(keys[i], nil, nil))
        }

        return sample
    }

    public func contains(_ x: [Any]) -> Bool {
        guard x.count == boxes.count else { return false }

        for (i, space) in boxes.enumerated() {
            if !space.contains(x[i]) {
                return false
            }
        }
        return true
    }

    public subscript(index: Int) -> any Space {
        return spaces[index]
    }
}
