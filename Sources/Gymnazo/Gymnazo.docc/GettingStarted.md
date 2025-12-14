# Getting Started

Learn how to create and interact with Gymnazo environments.

## Overview

Gymnazo follows the OpenAI Gymnasium API, making it familiar to anyone who has used the Python library.

## Creating an Environment

Use the `Gymnazo` registry to create environments by their ID:

```swift
import Gymnazo

// Create an environment
var env = Gymnazo.make("CartPole")
```

## The Environment Loop

Interact with environments using the standard `reset()` and `step(_:)` methods:

```swift
import MLX

// Reset the environment
var (observation, info) = env.reset()
var key = MLX.key(42)

var totalReward = 0.0
var done = false

while !done {
    // Sample a random action
    let action = env.action_space.sample(key: key)

    // Take a step
    let (nextObs, reward, terminated, truncated, stepInfo) = env.step(action)

    totalReward += reward
    observation = nextObs
    done = terminated || truncated
}

print("Episode finished with reward: \(totalReward)")
```

## Available Environments

Gymnazo includes several classic control and toy text environments:

| Environment           | ID                      | Description                 |
| --------------------- | ----------------------- | --------------------------- |
| CartPole              | `CartPole`              | Balance a pole on a cart    |
| MountainCar           | `MountainCar`           | Drive a car up a hill       |
| MountainCarContinuous | `MountainCarContinuous` | Continuous action version   |
| Acrobot               | `Acrobot`               | Swing up a two-link robot   |
| Pendulum              | `Pendulum`              | Swing up a pendulum         |
| FrozenLake            | `FrozenLake`            | Navigate a frozen lake grid |
| FrozenLake 8x8        | `FrozenLake8x8`         | Larger frozen lake variant  |

## Vector Environments

For parallel training, use vector environments to run multiple instances simultaneously.

The easiest way is with `make_vec(_:numEnvs:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:autoresetMode:kwargs:)`:

```swift
import Gymnazo

// Create 4 CartPole environments using make_vec
let vecEnv = Gymnazo.make_vec("CartPole", numEnvs: 4)

// Reset all environments at once
let (observations, _) = vecEnv.reset(seed: 42)
// observations.shape == [4, 4] for 4 envs with 4-dimensional observations

// Step all environments with batched actions
let result = vecEnv.step([1, 0, 1, 0])
```

For async execution, use `make_vec_async(_:numEnvs:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:autoresetMode:kwargs:)`:

```swift
let asyncEnv = Gymnazo.make_vec_async("CartPole", numEnvs: 4)

// Use stepAsync for parallel execution
let result = await asyncEnv.stepAsync([1, 0, 1, 0])
```

See <doc:VectorEnvironments> for more details.

## Default Wrappers

When you call `make(_:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:kwargs:)-(String,_,_,_,_,_,_,_)`, wrappers are applied automatically:

1. **PassiveEnvChecker** - Validates API compliance (disable with `disableEnvChecker: true`)
2. **OrderEnforcing** - Ensures `reset()` is called before `step(_:)`
3. **TimeLimit** - Truncates episodes at `maxEpisodeSteps` (if defined for the environment)

Note: `RecordEpisodeStatistics` is **not** applied by default. Enable it explicitly if you need episode tracking.

```swift
import Gymnazo

// Default wrappers applied
var env = Gymnazo.make("CartPole")

// Customize wrapper behavior
var env = Gymnazo.make(
    "CartPole",
    maxEpisodeSteps: 500,           // Override default time limit
    disableEnvChecker: true,        // Disable API validation
    recordEpisodeStatistics: true   // Enable statistics tracking
)

// Use maxEpisodeSteps: -1 to disable TimeLimit entirely
var env = Gymnazo.make("CartPole", maxEpisodeSteps: -1)
```

You can also apply wrappers manually using chainable extensions:

```swift
let env = CartPole()
    .validated()
    .recordingStatistics()
    .timeLimited(500)
```

See <doc:Wrappers-article> for the complete wrapper guide.
