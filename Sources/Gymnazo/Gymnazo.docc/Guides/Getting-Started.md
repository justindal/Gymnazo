# Getting Started

Learn how to create and interact with Gymnazo environments.

## Overview

Gymnazo follows the Gymnasium API, making it familiar to anyone who has used the Python library. All observations and actions are represented as `MLXArray` for seamless integration with MLX Swift.

## Creating an Environment

Use `Gymnazo` to create environments by their ID:

```swift
import Gymnazo
import MLX

var env = try await Gymnazo.make("CartPole")
```

### Environment Action Spaces

All environments use `MLXArray` for observations and actions. Actions from discrete spaces are scalar `MLXArray` values (use `.item(Int.self)` when needed):

| Environment | Action Space | Notes |
|-------------|--------------|-------|
| **Toy Text** | | |
| FrozenLake, FrozenLake8x8 | `Discrete(4)` | 0-3: Left, Down, Right, Up |
| Blackjack | `Discrete(2)` | 0: Stick, 1: Hit |
| Taxi | `Discrete(6)` | Movement + pickup/dropoff |
| CliffWalking | `Discrete(4)` | 0-3: Up, Right, Down, Left |
| **Classic Control** | | |
| CartPole | `Discrete(2)` | 0: Left, 1: Right |
| MountainCar | `Discrete(3)` | 0-2: Left, None, Right |
| MountainCarContinuous | `Box([-1], [1])` | Continuous force |
| Acrobot | `Discrete(3)` | Torque direction |
| Pendulum | `Box([-2], [2])` | Continuous torque |
| **Box2D** | | |
| LunarLander | `Discrete(4)` | Engine controls |
| LunarLanderContinuous | `Box` | Continuous engine |
| CarRacing | `Box` | Steering, gas, brake |
| CarRacingDiscrete | `Discrete(5)` | Discretized controls |

See <doc:Environments> to learn more about the included environments.

## The Environment Loop

Interact with environments using the standard `reset()` and `step(_:)` methods:

```swift
import Gymnazo
import MLX

var env = try await Gymnazo.make("CartPole")
let reset = try env.reset()
var observation = reset.obs
var key = MLX.key(42)

var totalReward = 0.0
var done = false

while !done {
    // Sample a random action (returns MLXArray)
    let (newKey, actionKey) = MLX.split(key: key)
    key = newKey
    let action = env.actionSpace.sample(key: actionKey)

    // Take a step
    let step = try env.step(action)

    totalReward += step.reward
    observation = step.obs
    done = step.terminated || step.truncated
}

print("Episode finished with reward: \(totalReward)")
```

## Vector Environments

For parallel training, use vector environments to run multiple instances simultaneously.

The easiest way is with `Gymnazo.makeVec(...)`:

```swift
import Gymnazo
import MLX

// Create 4 CartPole environments using makeVec
let vecEnv = try await Gymnazo.makeVec("CartPole", numEnvs: 4)

// Reset all environments at once
let reset = try vecEnv.reset(seed: 42)
// reset.observations.shape == [4, 4] for 4 envs with 4-dimensional observations

// Step all environments with batched actions (MLXArray)
let actions = MLXArray([1, 0, 1, 0])
let result = try vecEnv.step(actions)
```

For async execution, use `Gymnazo.makeVecAsync(...)`:

```swift
import Gymnazo
import MLX

let asyncEnv = try await Gymnazo.makeVecAsync("CartPole", numEnvs: 4)

// Use stepAsync for parallel execution
let actions = MLXArray([1, 0, 1, 0])
let result = try await asyncEnv.stepAsync(actions)
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
var env = try await Gymnazo.make("CartPole")

// Customize wrapper behavior
env = try await Gymnazo.make(
    "CartPole",
    maxEpisodeSteps: 500,           // Override default time limit
    disableEnvChecker: true,        // Disable API validation
    recordEpisodeStatistics: true   // Enable statistics tracking
)

// Use maxEpisodeSteps: -1 to disable TimeLimit entirely
env = try await Gymnazo.make("CartPole", maxEpisodeSteps: -1)
```

You can also apply wrappers manually using chainable extensions:

```swift
var env: any Env = try CartPole()
    .validated()
    .recordingStatistics()
    .timeLimited(500)
```

See <doc:Wrappers-Gym> for the complete wrapper guide.

## Training With Built-in Reinforcement Learning Algorithms

Gymnazo includes an (experimental) reinforcement learning module inspired by Stable-Baselines3.

See <doc:Reinforcement-Learning>.
