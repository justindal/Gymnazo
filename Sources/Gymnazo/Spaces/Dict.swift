//
//  Dict.swift
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

public struct Dict: Space {
    public typealias T = [String: Any]
    
    public let spaces: [String: any Space]
    private let boxes: [String: AnySpaceBox]
    
    public init(_ spaces: [String: any Space]) {
        self.spaces = spaces
        self.boxes = spaces.mapValues { makeBox($0) }
    }
    
    public var shape: [Int]? {
        return nil
    }
    
    public var dtype: DType? {
        return nil
    }
    
    public func sample(key: MLXArray, mask: MLXArray?, probability: MLXArray?) -> [String: Any] {
        let keys = MLX.split(key: key, into: boxes.count)
        var sample: [String: Any] = [:]
        
        let sortedKeys = boxes.keys.sorted()
        
        for (i, k) in sortedKeys.enumerated() {
            let space = boxes[k]!
            sample[k] = space.sample(keys[i], nil, nil)
        }
        
        return sample
    }
    
    public func contains(_ x: [String: Any]) -> Bool {
        for (k, space) in boxes {
            guard let value = x[k] else { return false }
            if !space.contains(value) {
                return false
            }
        }
        return true
    }
    
    public subscript(key: String) -> (any Space)? {
        return spaces[key]
    }
}
