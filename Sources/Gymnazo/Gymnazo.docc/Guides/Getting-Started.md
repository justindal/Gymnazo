# Getting Started

Learn how to create and interact with Gymnazo environments.

## Overview

Gymnazo follows the Gymnasium API, making it familiar to anyone who has used the Python library.

## Creating an Environment

Use the `Gymnazo` registry to create environments by their ID:

```swift
import Gymnazo

// Create an environment
var env = Gymnazo.make("CartPole")
```

See <doc:Environments> to learn more about the included environments.

## The Environment Loop

Interact with environments using the standard `reset()` and `step(_:)` methods:

```swift
import Gymnazo
import MLX

// Reset the environment
var env = Gymnazo.make("CartPole")
let reset = env.reset()
var observation = reset.obs
var key = MLX.key(42)

var totalReward = 0.0
var done = false

while !done {
    // Sample a random action
    let action = env.action_space.sample(key: key)

    // Take a step
    let step = env.step(action)

    totalReward += step.reward
    observation = step.obs
    done = step.terminated || step.truncated
}

print("Episode finished with reward: \(totalReward)")
```

## Vector Environments

For parallel training, use vector environments to run multiple instances simultaneously.

The easiest way is with `Gymnazo.make_vec(...)`:

```swift
import Gymnazo

// Create 4 CartPole environments using make_vec
let vecEnv = Gymnazo.make_vec("CartPole", numEnvs: 4)

// Reset all environments at once
let reset = vecEnv.reset(seed: 42)
// reset.observations.shape == [4, 4] for 4 envs with 4-dimensional observations

// Step all environments with batched actions
let result = vecEnv.step([1, 0, 1, 0])
```

For async execution, use `Gymnazo.make_vec_async(...)`:

```swift
import Gymnazo

let asyncEnv = Gymnazo.make_vec_async("CartPole", numEnvs: 4)

// Use stepAsync for parallel execution
let result = await asyncEnv.stepAsync([1, 0, 1, 0])
```

See <doc:Vector-Environments> for more details.

## Default Wrappers

When you call `Gymnazo.make(...)`, wrappers are applied automatically:

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

See <doc:Wrappers-Gym> for the complete wrapper guide.

## Training With Built-in Reinforcement Learning Algorithms

Gymnazo includes an (experimental) reinforcement learning module inspired by Stable-Baselines3.

See <doc:Reinforcement-Learning>.
