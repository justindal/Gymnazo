# ``Gymnazo``

A reinforcement learning toolkit written in Swift for Apple Platforms.

## Overview

Gymnazo provides a collection of environments and utilities for developing and testing reinforcement learning algorithms. Gymnazo also provides implementations of common reinforcement learning algorithms. Gymnazo is inspired by [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) and [Stable-baselines3](https://github.com/DLR-RM/stable-baselines3), and closely follows their APIs, for easier development. MLX is also used to benefit Apple Silicon devices. 

## Topics

### Introduction

- <doc:Getting-Started>

### Guides

- <doc:Core-Concepts>
- <doc:Creating-Environments>
- <doc:Reinforcement-Learning>
- <doc:Spaces>
- <doc:Wrappers-Gym>
- <doc:Vector-Environments>

### Environment Catalog

- <doc:Environments>

### Core Protocols

- ``Env``
- ``Space``
- ``Wrapper``

### Space Types (API)

- ``Discrete``
- ``Box``
- ``MultiDiscrete``
- ``MultiBinary``
- ``TextSpace``
- ``Tuple``
- ``Dict``
- ``SequenceSpace``
- ``Graph``

### Vector Environment Types

- ``VectorEnv``
- ``SyncVectorEnv``
- ``AsyncVectorEnv``
- ``AutoresetMode``
- ``VectorStepResult``
- ``VectorResetResult``

### Wrappers

- ``TimeLimit``
- ``OrderEnforcing``
- ``PassiveEnvChecker``
- ``RecordEpisodeStatistics``
- ``AutoReset``
- ``TransformObservation``
- ``NormalizeObservation``
- ``FlattenObservation``
- ``TransformReward``
- ``NormalizeReward``   
- ``ClipAction``
- ``RescaleAction``

### Reinforcement Learning

- ``DQN``
- ``SAC``
- ``TabularAgent``

### Registration

- ``Gymnazo``
- ``EnvSpec``
- ``EnvOptions``
