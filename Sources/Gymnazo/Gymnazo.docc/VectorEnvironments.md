# Vector Environments

Run multiple environment instances in parallel for efficient training.

## Overview

Vector environments allow you to run multiple instances of an environment simultaneously, which can help speed up training. Gymnazo provides two implementations, as Gymnasium does:

- **SyncVectorEnv**: Runs environments sequentially in the main thread.
- **AsyncVectorEnv**: Runs each environment in its own Swift actor for true parallelism.

## Creating Vector Environments

### Using make_vec() (Recommended)

The simplest way to create vector environments is with `make_vec(_:numEnvs:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:autoresetMode:kwargs:)` and `make_vec_async(_:numEnvs:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:autoresetMode:kwargs:)`:

```swift
import Gymnazo

// Synchronous vector environment with 4 CartPoles
let syncEnv = Gymnazo.make_vec("CartPole", numEnvs: 4)

// Asynchronous vector environment with 4 CartPoles
let asyncEnv = Gymnazo.make_vec_async("CartPole", numEnvs: 4)
```

You can also use the `vectorizationMode` parameter:

```swift
// Equivalent to make_vec_async
let asyncEnv = Gymnazo.make_vec("CartPole", numEnvs: 4, vectorizationMode: .async)
```

These functions apply the same default wrappers as `make(_:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:kwargs:)-(String,_,_,_,_,_,_,_)` to each sub-environment.

### Using Factory Closures

For more control, create vector environments with factory closures:

```swift
import Gymnazo

// Synchronous vector environment
let syncEnv = SyncVectorEnv(envFns: [
    { CartPole() },
    { CartPole() },
    { CartPole() },
    { CartPole() }
])

// Asynchronous vector environment
let asyncEnv = AsyncVectorEnv(envFns: [
    { CartPole() },
    { CartPole() },
    { CartPole() },
    { CartPole() }
])
```

You can also use `make_vec(envFns:autoresetMode:)` with custom factory closures:

```swift
// Environments with different configurations
let envs = Gymnazo.make_vec(envFns: [
    { Gymnazo.make("Pendulum", kwargs: ["g": 9.81]) },
    { Gymnazo.make("Pendulum", kwargs: ["g": 1.62]) },  // Moon gravity
])
```

## Basic Usage

### Reset

Reset returns batched observations from all environments:

```swift
let (observations, infos) = syncEnv.reset(seed: 42)
// observations.shape == [num_envs, obs_dim]
// For CartPole with 4 envs: [4, 4]
```

### Step

Step takes an array of actions (one per environment) and returns batched results:

```swift
let actions = [1, 0, 1, 0]  // One action per environment
let result = syncEnv.step(actions)

// Access results
result.observations  // [num_envs, obs_dim]
result.rewards       // [num_envs]
result.terminations  // [num_envs] - Bool array
result.truncations   // [num_envs] - Bool array
result.infos         // Dictionary with additional info
```

## Autoreset Modes

When a sub-environment terminates, vector environments can handle the reset automatically. Control this behavior with `AutoresetMode`:

### Next Step (Default)

The environment resets on the _next_ step call. The final observation is preserved in the info dictionary:

```swift
let env = SyncVectorEnv(
    envFns: envFactories,
    autoresetMode: .nextStep  // Default
)

let result = env.step(actions)

// If environment 2 terminated, access its final observation:
if let finalObs = result.infos["final_observation"] as? [MLXArray?] {
    let env2FinalObs = finalObs[2]
}
```

### Same Step

The environment resets immediately in the same step. The returned observation is from the new episode:

```swift
let env = SyncVectorEnv(
    envFns: envFactories,
    autoresetMode: .sameStep
)
```

### Disabled

No automatic reset. You must manually call `reset(seed:options:)` when environments terminate:

```swift
let env = SyncVectorEnv(
    envFns: envFactories,
    autoresetMode: .disabled
)
```

## Async Vector Environment

`AsyncVectorEnv` uses Swift's actor system to run environments in parallel. It provides both synchronous and asynchronous APIs:

```swift
let asyncEnv = Gymnazo.make_vec_async("CartPole", numEnvs: 4)

// Synchronous API (runs parallel internally but blocks)
let (obs, _) = asyncEnv.reset(seed: 42)
let result = asyncEnv.step([1, 0, 1, 0])

// Asynchronous API (truly non-blocking)
let (obs, _) = await asyncEnv.resetAsync(seed: 42)
let result = await asyncEnv.stepAsync([1, 0, 1, 0])
```

### When to Use Async

Use `AsyncVectorEnv` when:

- Your environment's `step(_:)` function is computationally expensive
- You have many environments and want true parallelism
- You're running on a multi-core system

Use `SyncVectorEnv` when:

- Environments are lightweight
- You need deterministic, reproducible behavior
- You want simpler debugging

## Properties

Both vector environments expose useful properties:

```swift
env.num_envs                  // Number of sub-environments
env.single_observation_space  // Observation space of one env
env.single_action_space       // Action space of one env
```

## Example: Training Loop

```swift
import Gymnazo
import MLX

@MainActor
func trainWithVectorEnv() {
    let numEnvs = 8

    // Use make_vec for the simplest setup
    let env = Gymnazo.make_vec("CartPole", numEnvs: numEnvs)

    var (observations, _) = env.reset(seed: 42)
    var key = MLX.key(0)

    for step in 0..<10000 {
        // Sample random actions for all environments
        var actions: [Int] = []
        for _ in 0..<numEnvs {
            let action = env.single_action_space.sample(key: key).item(Int.self)
            actions.append(action)
            key = MLX.split(key: key).0
        }

        // Step all environments
        let result = env.step(actions)
        observations = result.observations

        // Process rewards, check terminations, etc.
        let totalReward = result.rewards.sum().item(Float.self)
        print("Step \(step): Total reward = \(totalReward)")
    }
}
```

## Topics

### Vector Environment Types

- `VectorEnv`
- `SyncVectorEnv`
- `AsyncVectorEnv`

### Configuration

- `AutoresetMode`
- `VectorizationMode`

### Result Types

- `VectorStepResult`
- `VectorResetResult`
- `EnvStepResult`
- `EnvResetResult`
