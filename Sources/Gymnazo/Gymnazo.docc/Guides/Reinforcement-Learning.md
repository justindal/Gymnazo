# Reinforcement Learning

Gymnazo includes a reinforcement learning module inspired by Stable-Baselines3.

## Overview

The RL module provides ready-to-use algorithm implementations that work with any Gymnazo environment. All algorithms are Swift actors, making them safe for concurrent use and easy to integrate with SwiftUI apps via callbacks.

Five algorithms are supported:

| Algorithm | Type | Action Space | Use Case |
|-----------|------|--------------|----------|
| ``DQN`` | Deep Q-Network | Discrete | CartPole, LunarLander, Atari-style |
| ``SAC`` | Soft Actor-Critic | Continuous | Pendulum, MountainCarContinuous, robotics |
| ``PPO`` | Proximal Policy Optimization | Discrete + Continuous + MultiDiscrete + MultiBinary | General-purpose on-policy training |
| ``TD3`` | Twin Delayed DDPG | Continuous | MountainCarContinuous, Pendulum, robotics |
| ``TabularAgent`` | Q-Learning / SARSA | Discrete (small) | FrozenLake, Taxi, CliffWalking |

## DQN (Deep Q-Network)

``DQN`` is an off-policy algorithm for discrete action spaces. It uses a replay buffer and a target network for stable training.

```swift
import Gymnazo

let env = try await Gymnazo.make("CartPole")

let model = try DQN(env: env)

try await model.learn(totalTimesteps: 50_000, callbacks: nil)
```

Configure DQN with ``DQNConfig``:

```swift
let model = try DQN(
    env: env,
    learningRate: ConstantLearningRate(1e-4),
    config: DQNConfig(
        batchSize: 64,
        gamma: 0.99,
        explorationFraction: 0.2,
        explorationFinalEps: 0.05
    )
)
```

## SAC (Soft Actor-Critic)

``SAC`` is an off-policy algorithm for continuous action spaces. It maximizes both reward and entropy for robust exploration.

```swift
import Gymnazo

let env = try await Gymnazo.make("Pendulum")

let model = SAC(env: env)

try await model.learn(totalTimesteps: 100_000, callbacks: nil)
```

Configure SAC with ``OffPolicyConfig`` and ``SACNetworksConfig``:

```swift
let model = SAC(
    env: env,
    learningRate: ConstantLearningRate(3e-4),
    config: OffPolicyConfig(
        batchSize: 256,
        gamma: 0.99,
        learningStarts: 1000
    ),
    entCoef: .auto()
)
```

## PPO (Proximal Policy Optimization)

``PPO`` is an on-policy actor-critic algorithm that collects rollout trajectories and optimizes a clipped surrogate objective.

```swift
import Gymnazo

let env = try await Gymnazo.make("CartPole")

let model = try PPO(
    env: env,
    learningRate: ConstantLearningRate(3e-4),
    policyConfig: PPOPolicyConfig(featuresExtractor: .flatten),
    config: PPOConfig(
        nSteps: 128,
        batchSize: 32,
        nEpochs: 4,
        gamma: 0.99,
        gaeLambda: 0.95,
        clipRange: 0.2
    )
)

try await model.learn(totalTimesteps: 100_000, callbacks: nil)
```

## TD3 (Twin Delayed DDPG)

``TD3`` is an off-policy algorithm for continuous action spaces. It improves on DDPG by using two critic networks and a delayed actor update to reduce overestimation bias.

```swift
import Gymnazo

let env = try await Gymnazo.make("Pendulum")

let model = TD3(env: env)

try await model.learn(totalTimesteps: 100_000, callbacks: nil)
```

Configure TD3 with ``OffPolicyConfig``, ``TD3PolicyConfig``, and ``TD3AlgorithmConfig``:

```swift
let model = TD3(
    env: env,
    learningRate: ConstantLearningRate(1e-3),
    policyConfig: TD3PolicyConfig(
        netArch: nil,       // uses [400, 300] defaults
        nCritics: 2
    ),
    algorithmConfig: TD3AlgorithmConfig(
        policyDelay: 2,
        targetPolicyNoise: 0.2,
        targetNoiseClip: 0.5,
        actionNoise: .normal(std: 0.1)
    ),
    config: OffPolicyConfig(
        batchSize: 256,
        gamma: 0.99,
        learningStarts: 1000
    )
)
```

## Tabular Agents (Q-Learning / SARSA)

``TabularAgent`` implements classic tabular methods for small discrete state and action spaces. Choose between Q-Learning (off-policy) and SARSA (on-policy) via ``TabularAgent/UpdateRule``.

```swift
import Gymnazo

let env = try await Gymnazo.make("FrozenLake", options: ["isSlippery": false])

let agent = TabularAgent(
    env: env,
    updateRule: .qLearning,
    config: TabularConfig(
        learningRate: 0.1,
        gamma: 0.99,
        epsilon: 1.0,
        epsilonDecay: 0.995,
        minEpsilon: 0.01
    )
)

try await agent.learn(totalTimesteps: 100_000, callbacks: nil)
```

## Callbacks

All algorithms accept ``LearnCallbacks`` for monitoring training progress and controlling execution. This is useful for updating UI during training.

```swift
let callbacks = LearnCallbacks(
    onStep: { currentTimestep, totalTimesteps, _ in
        // Return false to stop training early
        return true
    },
    onEpisodeEnd: { reward, length in
        print("Episode finished: reward=\(reward), length=\(length)")
    },
    onTrain: { metrics in
        if let loss = metrics["loss"] {
            print("Loss: \(loss)")
        }
    }
)

try await model.learn(totalTimesteps: 50_000, callbacks: callbacks)
```

Similarly, ``EvaluateCallbacks`` can be used with the `evaluate()` method. Note that its `onStep` closure takes no arguments:

```swift
let evalCallbacks = EvaluateCallbacks(
    onStep: {
        // Return false to stop evaluation early
        return true
    },
    onEpisodeEnd: { reward, length in
        print("Episode: reward=\(reward), length=\(length)")
    }
)

try await model.evaluate(episodes: 10, deterministic: true, callbacks: evalCallbacks)
```

## Persistence

All algorithms support saving and loading checkpoints, including model weights, configuration, and optionally the replay buffer.

### Saving

```swift
let saveURL = URL.documentsDirectory.appending(path: "my-agent")
try await model.save(to: saveURL)
```

### Loading

```swift
let model = try DQN.load(from: saveURL, env: env)

// For deep RL, optionally skip the replay buffer
let model = try SAC.load(from: saveURL, includeBuffer: false)
let model = try PPO.load(from: saveURL, env: env)

// Tabular agents
let agent = try TabularAgent.load(from: saveURL, env: env)
```

### Resuming Training

```swift
var model = try DQN.load(from: saveURL, env: env)
try await model.learn(totalTimesteps: 50_000, callbacks: nil, resetProgress: false)
```

## Topics

### Algorithms

- ``DQN``
- ``SAC``
- ``PPO``
- ``TD3``
- ``TabularAgent``

### Algorithm Configuration

- ``DQNConfig``
- ``DQNPolicyConfig``
- ``DQNOptimizerConfig``
- ``OffPolicyConfig``
- ``TrainFrequency``
- ``TrainFrequencyUnit``
- ``GradientSteps``
- ``SACNetworksConfig``
- ``SACActorConfig``
- ``SACCriticConfig``
- ``SACOptimizerConfig``
- ``EntropyCoef``
- ``PPOConfig``
- ``PPOPolicyConfig``
- ``TD3PolicyConfig``
- ``TD3AlgorithmConfig``
- ``TD3ActorConfig``
- ``TD3ActionNoiseConfig``
- ``TabularConfig``

### Core Protocols

- ``Model``
- ``Policy``
- ``FeaturesExtractor``
- ``Distribution``

### Networks

- ``FlattenExtractor``
- ``NatureCNN``
- ``CombinedExtractor``
- ``MLPExtractor``
- ``NetArch``

### Buffers

- ``Buffer``
- ``ReplayBuffer``
- ``RolloutBuffering``
- ``RolloutBuffer``

### Persistence

- ``AlgorithmCheckpoint``
- ``AlgorithmKind``

### Callbacks and Schedules

- ``LearnCallbacks``
- ``EvaluateCallbacks``
- ``LearningRateSchedule``
- ``ConstantLearningRate``
