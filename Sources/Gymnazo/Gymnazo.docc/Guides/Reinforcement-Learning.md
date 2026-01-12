# Reinforcement Learning

This section is incomplete.

Gymnazo includes an (experimental) reinforcement learning module inspired by Stable-Baselines3.

## Supported algorithms

- ``SAC`` (Soft Actor-Critic): off-policy continuous-control for `MLXArray` observations and actions.

## SAC Training Example

Example (``Pendulum``):

```swift
import Gymnazo
import MLX

let env = Gymnazo.make("Pendulum")

let model = SAC(
    observationSpace: env.observation_space,
    actionSpace: env.action_space,
    env: env
)

model.learn(totalTimesteps: 100_000)
```

## Topics

### Algorithms

- ``Algorithm``
- ``OffPolicyAlgorithm``
- ``SAC``

### Configurations
- ``OffPolicyConfig``
- ``TrainFrequency``
- ``TrainFrequencyUnit``
- ``GradientSteps``

### Data

- ``ReplayBuffer``
