//
//  ActorCriticNetArch.swift
//  Gymnazo
//
//

import MLXNN

/// Architecture specification for off-policy actor-critic algorithms (SAC/TD3/DDPG).
/// A list means actor and critic share the same hidden sizes and a dict means separate
///  sizes for actor `pi`. and critic `qf`.
public enum ActorCriticNetArch {
    case shared([Int])

    case separate(pi: [Int], qf: [Int])
}

