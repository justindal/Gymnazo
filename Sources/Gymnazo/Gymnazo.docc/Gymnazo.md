# `Gymnazo`

A Swift implementation of Farama's Gymnasium for reinforcement learning on Apple platforms.

## Overview

Gymnazo provides a collection of environments and utilities for developing and testing reinforcement learning algorithms. It includes built-in implementations of common RL algorithms like Q-Learning, SARSA, DQN, and SAC, so you can start training agents right away.

## Topics

### Getting Started

- <doc:GettingStarted>

### Core Protocols

- `Env`
- `Space`
- `Wrapper`

### RL Agent Protocols

- `DiscreteRLAgent`
- `DiscreteDeepRLAgent`
- `ContinuousDeepRLAgent`

### Environments

- `CartPole`
- `MountainCar`
- `MountainCarContinuous`
- `Acrobot`
- `FrozenLake`

### Spaces

- `Discrete`
- `Box`
- `MultiDiscrete`
- `Dict`
- `Tuple`

### RL Agent Implementations

- `QLearning`
- `SARSA`
- `DQN`
- `SAC`
- `DiscreteAgent`

### Wrappers

- `TimeLimit`
- `OrderEnforcing`
- `PassiveEnvChecker`
- `RecordEpisodeStatistics`
- `TransformObservation`
- `NormalizeObservation`

### Registration

- `EnvSpec`
- `GymnazoRegistrations`
