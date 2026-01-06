//
//  Algorithm.swift
//  Gymnazo
//
//  Created by Justin Daludado on 2026-01-06.
//

protocol Algorithm {
    var policy: String { get }
    var env: Env { get }
    var learningRate: Double { get }
    
    
}
