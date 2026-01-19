# Vector Environments

Run multiple environment instances in parallel for efficient training.

## Overview

Vector environments allow you to run multiple instances of an environment simultaneously, which can help speed up training. Gymnazo provides two implementations, as Gymnasium does:

- **SyncVectorEnv**: Runs environments sequentially in the main thread.
- **AsyncVectorEnv**: Runs each environment in its own Swift actor for true parallelism.

## Creating Vector Environments

### Using makeVec() (Recommended)

The simplest way to create vector environments is with `Gymnazo.makeVec(...)` and `Gymnazo.makeVecAsync(...)`:

```swift
import Gymnazo

// Synchronous vector environment with 4 CartPoles
let syncEnv: SyncVectorEnv<Int> = try await Gymnazo.makeVec("CartPole", numEnvs: 4)

// Asynchronous vector environment with 4 CartPoles
let asyncEnv: AsyncVectorEnv<Int> = try await Gymnazo.makeVecAsync("CartPole", numEnvs: 4)
```

You can also use the `vectorizationMode` parameter:

```swift
// Equivalent to makeVecAsync
let asyncEnv: any VectorEnv<Int> = try await Gymnazo.makeVec(
    "CartPole",
    numEnvs: 4,
    vectorizationMode: .async
)
```

These functions apply the same default wrappers as `Gymnazo.make(...)` to each sub-environment.

### Using Factory Closures

For more control, create vector environments with factory closures:

```swift
import Gymnazo

// Synchronous vector environment
let syncEnv: SyncVectorEnv<Int> = try SyncVectorEnv(envFns: [
    { CartPole() },
    { CartPole() },
    { CartPole() },
    { CartPole() }
])

// Asynchronous vector environment
let asyncEnv: AsyncVectorEnv<Int> = try AsyncVectorEnv(envFns: [
    { CartPole() },
    { CartPole() },
    { CartPole() },
    { CartPole() }
])
```

You can also use `makeVec(envFns:autoresetMode:)` with custom factory closures:

```swift
// Environments with different configurations
let envs = try await Gymnazo.makeVec(envFns: [
    { Pendulum(g: 9.81) },
    { Pendulum(g: 1.62) },  // Moon gravity
])
```

## Basic Usage

### Reset

Reset returns batched observations from all environments:

```swift
let reset = try syncEnv.reset(seed: 42)
let observations = reset.observations
let infos = reset.infos

// observations.shape == [numEnvs, obsDim]
// For CartPole with 4 envs: [4, 4]
```

### Step

Step takes an array of actions (one per environment) and returns batched results:

```swift
let actions = [1, 0, 1, 0]  // One action per environment
let result = try syncEnv.step(actions)

// Access results
result.observations  // [numEnvs, obsDim]
result.rewards       // [numEnvs]
result.terminations  // [numEnvs] - Bool array
result.truncations   // [numEnvs] - Bool array
result.infos         // [Info] with additional info
```

## Autoreset Modes

When a sub-environment terminates, vector environments can handle the reset automatically. Control this behavior with `AutoresetMode`:

### Next Step (Default)

The environment resets on the _next_ step call. The final observation is preserved in the info dictionary:

```swift
let env = try SyncVectorEnv(
    envFns: envFactories,
    autoresetMode: .nextStep  // Default
)

let result = try env.step(actions)

// If any sub-environment finished, access terminal infos:
for (index, info) in result.infos.enumerated() {
    if let finalInfo = info["final_info"]?.object {
        let terminalObs = info["final_observation"]?.cast(MLXArray.self)
        _ = (index, terminalObs, finalInfo)
    }
}
```

### Same Step

The environment resets immediately in the same step. The returned observation is from the new episode:

```swift
let env = try SyncVectorEnv(
    envFns: envFactories,
    autoresetMode: .sameStep
)
```

### Disabled

No automatic reset. You must manually call `reset(seed:options:)` when environments terminate:

```swift
let env = try SyncVectorEnv(
    envFns: envFactories,
    autoresetMode: .disabled
)
```

## Async Vector Environment

`AsyncVectorEnv` uses Swift's actor system to run environments in parallel. It provides both synchronous and asynchronous APIs:

```swift
let asyncEnv: AsyncVectorEnv<Int> = try await Gymnazo.makeVecAsync("CartPole", numEnvs: 4)

// Synchronous API (runs parallel internally but blocks)
let reset = try asyncEnv.reset(seed: 42)
let result = try asyncEnv.step([1, 0, 1, 0])

// Asynchronous API (truly non-blocking)
let reset = try await asyncEnv.resetAsync(seed: 42)
let result = try await asyncEnv.stepAsync([1, 0, 1, 0])
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
env.numEnvs                  // Number of sub-environments
env.singleObservationSpace   // Observation space of one env
env.singleActionSpace        // Action space of one env
```

## Example: Training Loop

```swift
import Gymnazo
import MLX

@MainActor
func trainWithVectorEnv() async throws {
    let numEnvs = 8

    // Use makeVec for the simplest setup
    let env: SyncVectorEnv<Int> = try await Gymnazo.makeVec("CartPole", numEnvs: numEnvs)

    var observations = try env.reset(seed: 42).observations
    var key = MLX.key(0)
    let actionSpace = env.singleActionSpace as! Discrete

    for step in 0..<10000 {
        // Sample random actions for all environments
        var actions: [Int] = []
        for _ in 0..<numEnvs {
            let action = actionSpace.sample(key: key)
            actions.append(action)
            key = MLX.split(key: key).0
        }

        // Step all environments
        let result = try env.step(actions)
        observations = result.observations

        // Process rewards, check terminations, etc.
        let totalReward = result.rewards.sum().item(Float.self)
        print("Step \(step): Total reward = \(totalReward)")
    }
}
```

## Topics

### Vector Environment Types

- ``VectorEnv``
- ``SyncVectorEnv``
- ``AsyncVectorEnv``

### Configuration

- ``AutoresetMode``
- ``VectorizationMode``

### Result Types

- ``VectorStepResult``
- ``VectorResetResult``
- ``EnvStepResult``
- ``EnvResetResult``
