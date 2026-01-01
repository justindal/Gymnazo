# ``Gymnazo``

A Swift implementation of Farama's Gymnasium for reinforcement learning on Apple platforms.

## Overview

Gymnazo provides a collection of environments and utilities for developing and testing reinforcement learning algorithms. Built with MLX Swift for Apple Silicon acceleration.

## Topics

### Getting Started

- <doc:GettingStarted>

### Guides

- <doc:VectorEnvironments>
- <doc:Wrappers-article>

### Core Protocols

- ``Env``
- ``Space``
- ``Wrapper``

### Creating Environments

- ``make(_:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:kwargs:)-(String,_,_,_,_,_,_,_)``
- ``make_vec(_:numEnvs:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:autoresetMode:kwargs:)``
- ``make_vec_async(_:numEnvs:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:autoresetMode:kwargs:)``
- ``register(id:entryPoint:maxEpisodeSteps:rewardThreshold:nondeterministic:)``

### Environments

- ``CartPole``
- ``MountainCar``
- ``MountainCarContinuous``
- ``Acrobot``
- ``Pendulum``
- ``FrozenLake``
- ``LunarLander``
- ``LunarLanderContinuous``
- ``CarRacing``
- ``CarRacingDiscrete``

### Environment Guides

- <doc:CartPole>
- <doc:MountainCar>
- <doc:MountainCarContinuous>
- <doc:Acrobot>
- <doc:Pendulum>
- <doc:FrozenLake>
- <doc:Taxi>
- <doc:CliffWalking>
- <doc:Blackjack>
- <doc:LunarLander>
- <doc:LunarLanderContinuous>
- <doc:CarRacing>

### Spaces

- ``Discrete``
- ``Box``
- ``MultiDiscrete``
- ``MultiBinary``
- ``TextSpace``
- ``Tuple``
- ``Dict``
- ``SequenceSpace``
- ``Graph``

### Vector Environments

- ``VectorEnv``
- ``SyncVectorEnv``
- ``AsyncVectorEnv``
- ``AutoresetMode``
- ``VectorStepResult``

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

### Registration

- ``EnvSpec``
- ``GymnazoRegistrations``
