//
//  Keys.swift
//  Gymnazo
//
//  Created by Justin Daludado on 2026-02-07.
//

import MLX

public func nextKey(for key: inout MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    let (newKey, useKey) = MLX.split(key: key, stream: stream)
    key = newKey
    return useKey
}
