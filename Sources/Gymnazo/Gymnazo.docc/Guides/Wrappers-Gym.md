# Wrappers

Modify environment behavior using composable wrappers.

## Overview

Wrappers allow you to modify an environment's behavior without changing its underlying implementation. They follow the decorator pattern—each wrapper wraps an environment (or another wrapper) and intercepts calls to `step(_:)`, `reset(seed:options:)`, and other methods.

## Default Wrappers via make()

When you call `Gymnazo.make(...)`, wrappers are applied automatically in this order:

1. **PassiveEnvChecker** - Validates API compliance (disable with `disableEnvChecker: true`)
2. **OrderEnforcing** - Ensures `reset(seed:options:)` is called before `step(_:)`
3. **TimeLimit** - Truncates episodes at `maxEpisodeSteps` (if the environment defines one)

Note: `RecordEpisodeStatistics` is **not** applied by default. Enable it explicitly if needed.

```swift
import Gymnazo

// Create environment with default wrappers
var env = try await Gymnazo.make("CartPole")

// Customize wrapper behavior
env = try await Gymnazo.make(
    "CartPole",
    maxEpisodeSteps: 500,           // Override default time limit
    disableEnvChecker: true,        // Disable PassiveEnvChecker
    recordEpisodeStatistics: true   // Enable RecordEpisodeStatistics
)

// Use maxEpisodeSteps: -1 to disable TimeLimit entirely
env = try await Gymnazo.make("CartPole", maxEpisodeSteps: -1)
```

## Chainable Wrapper Extensions

Gymnazo provides chainable extension methods for applying wrappers beyond what `Gymnazo.make(...)` provides:

```swift
import Gymnazo

// Start with make() for default wrappers, then add extras
var env = try await Gymnazo.make("MountainCarContinuous")
env = try env
    .actionsRescaled(from: (-1.0, 1.0))  // Rescale agent outputs
    .observationsNormalized()             // Normalize observations

// Or for custom setups without make()
var customEnv: any Env = try CartPole()
    .validated(maxSteps: 500)       // Standard wrapper stack
    .observationsTransformed { $0 / 10.0 }  // Custom normalization
```

### Wrapper Order

Wrappers are applied inside-out—the **last wrapper in the chain processes calls first**.

```swift
// Example: Adding observation normalization to an environment
var env = try await Gymnazo.make("CartPole")
env = try env
    .observationsNormalized()  // Applied after make()'s wrappers
```

Call flow: `Your code → NormalizeObservation → [make() wrappers] → CartPole`

> **Tip:** Use `.validated(maxSteps:)` when not using `make()` to apply the standard stack (PassiveEnvChecker + OrderEnforcing + TimeLimit):
>
> ```swift
> var env: any Env = try CartPole()
>     .validated(maxSteps: 500)
>     .observationsNormalized()
> ```

### Available Extension Methods

| Method                         | Wrapper                            | Description                   |
| ------------------------------ | ---------------------------------- | ----------------------------- |
| `.passiveChecked()`            | `PassiveEnvChecker`                | Validates API compliance      |
| `.orderEnforced()`             | `OrderEnforcing`                   | Ensures reset() before step() |
| `.timeLimited(_:)`             | `TimeLimit`                        | Limits episode steps          |
| `.recordingStatistics()`       | `RecordEpisodeStatistics`          | Tracks episode metrics        |
| `.observationsNormalized()`    | `NormalizeObservation`             | Normalizes observations       |
| `.observationsTransformed(_:)` | `TransformObservation`             | Custom observation transform  |
| `.observationsFlattened()`     | `FlattenObservation`               | Flattens observations         |
| `.rewardsTransformed(_:)`      | `TransformReward`                  | Custom reward transform       |
| `.rewardsNormalized()`         | `NormalizeReward`                  | Normalizes rewards            |
| `.autoReset(mode:)`            | `AutoReset`                        | Resets episodes automatically |
| `.actionsClipped()`            | `ClipAction`                       | Clips actions to bounds       |
| `.actionsRescaled(from:)`      | `RescaleAction`                    | Rescales action range         |
| `.validated()`                 | PassiveEnvChecker + OrderEnforcing | Standard validation stack     |
| `.validated(maxSteps:)`        | + TimeLimit                        | Validation with time limit    |

## Manual Wrapper Application

You can also use constructor syntax directly:

```swift
import Gymnazo

var base = CartPole()
var wrapped: any Env = try TimeLimit(env: base, maxEpisodeSteps: 500)
```

## Available Wrappers

### TimeLimit

Truncates episodes after a maximum number of steps:

```swift
import Gymnazo
import MLX

var env: any Env = try TimeLimit(env: CartPole(), maxEpisodeSteps: 200)

// After 200 steps, truncated will be true
let _ = try env.reset()
let step = try env.step(MLXArray(0))
let truncated = step.truncated
```

### RecordEpisodeStatistics

Tracks episode returns, lengths, and timing. **Not applied by default** - enable with `recordEpisodeStatistics: true`.

```swift
// Enable via make()
var env = try await Gymnazo.make("CartPole", recordEpisodeStatistics: true)

// Or apply manually
var statsEnv = try RecordEpisodeStatistics(env: CartPole(), bufferLength: 100)

// Run some episodes...

// Access statistics (stored in queues)
statsEnv.returnQueue   // Last 100 episode returns
statsEnv.lengthQueue   // Last 100 episode lengths
statsEnv.timeQueue     // Last 100 episode durations
statsEnv.episodeCount  // Total episodes completed
```

When an episode ends, the statistics are also added to the step's info dictionary under the `"episode"` key:

```swift
import MLX

let action = MLXArray(0)
let result = try env.step(action)
if result.terminated || result.truncated {
    if let stats = result.info["episode"]?.object {
        let episodeReturn = stats["r"]?.double
        let episodeLength = stats["l"]?.int
        let episodeTime = stats["t"]?.double
    }
}
```

### OrderEnforcing

Ensures `reset(seed:options:)` is called before `step(_:)`. Applied by default via `Gymnazo.make(...)`.

```swift
import Gymnazo
import MLX

var env: any Env = OrderEnforcing(env: CartPole())
let action = MLXArray(0)

// This will trigger an assertion failure:
// env.step(action)  // Error: reset() not called

let _ = try env.reset()
let _ = try env.step(action)  // OK
```

### PassiveEnvChecker

Validates environment API compliance during runtime. Applied by default via `Gymnazo.make(...)`.

```swift
var env: any Env = PassiveEnvChecker(env: CartPole())

// Automatically checks:
// - Observation matches observationSpace
// - Action is valid for actionSpace
// - Return types are correct
```

Disable with `Gymnazo.make("CartPole", disableEnvChecker: true)`.

### TransformObservation

Applies a custom transformation to observations:

```swift
var env: any Env = TransformObservation(
    env: CartPole(),
    transform: { obs in
        // Normalize observation
        return obs / 10.0
    }
)
```

### NormalizeObservation

Normalizes observations using running mean and standard deviation:

```swift
var env: any Env = try NormalizeObservation(env: CartPole())

// Observations are automatically normalized to approximately N(0, 1)
```

### ClipAction

Clips continuous actions to the action space bounds:

```swift
import MLX

var env: any Env = ClipAction(env: MountainCarContinuous())

// Actions outside [-1, 1] are automatically clipped
let clippedStep = try env.step(MLXArray([2.0]))  // Clipped to 1.0
```

### RescaleAction

Rescales actions from a source range (default `[-1, 1]`) to the environment's action space bounds:

```swift
var env: any Env = RescaleAction(
    env: MountainCarContinuous(),
    sourceLow: -1.0,
    sourceHigh: 1.0
)

// Your agent outputs [-1, 1], wrapper rescales to env's actual range
```

## Wrapper Chaining

When applying wrappers manually, they chain inside-out—the last wrapper applied is the first to handle calls:

```swift
import Gymnazo

var base = CartPole()
let checked = PassiveEnvChecker(env: base)
let limited = try TimeLimit(env: checked, maxEpisodeSteps: 500)
var recorded: any Env = try RecordEpisodeStatistics(env: limited)
// Use `recorded` - it's the outermost wrapper
```

Order matters! In this example:

1. `RecordEpisodeStatistics` sees the step first
2. `TimeLimit` may truncate the episode
3. `PassiveEnvChecker` validates the underlying call

## Accessing the Base Environment

Every `Env` has an `Env/unwrapped` property that returns the base environment:

```swift
let baseEnv = env.unwrapped  // Returns the innermost, unwrapped environment
```

## Custom Wrappers

Create your own wrapper by conforming to the `Wrapper` protocol:

```swift
import Gymnazo
import MLX

public final class RewardScaler: Wrapper {
    public var env: any Env
    public let scale: Double

    public required init(env: any Env) {
        self.env = env
        self.scale = 1.0
    }

    public init(env: any Env, scale: Double) {
        self.env = env
        self.scale = scale
    }

    public func step(_ action: MLXArray) throws -> Step {
        var result = try env.step(action)
        return Step(
            obs: result.obs,
            reward: result.reward * scale,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info
        )
    }

    // reset() uses the default pass-through from Wrapper protocol
}

// Usage
var env: any Env = RewardScaler(env: CartPole(), scale: 0.01)
```

### FlattenObservation

Flattens observations into a 1D `MLXArray` and updates the observation space to a flat `Box`. This wrapper only works when `flatten_space(env.observationSpace)` returns a `Box`. For `SequenceSpace` and `Graph`, `flatten_space` preserves the structured space instead of forcing a single `Box`.

```swift
import Gymnazo

var env: any Env = try FrozenLake().observationsFlattened()
let reset = try env.reset(seed: 0)
let obs = reset.obs
```

### TransformReward

Applies a custom function to each reward.

```swift
import Gymnazo

var env: any Env = try CartPole().rewardsTransformed { $0 * 2 }
```

### NormalizeReward

Normalizes rewards using a running return variance estimate.

```swift
import Gymnazo

var env: any Env = try CartPole().rewardsNormalized()
```

### AutoReset

Automatically resets when an episode ends. Two modes are available:

- **`.sameStep`**: Resets immediately in the same step. Returns the new episode's observation, with the final observation stored in `info["final_observation"]`.
- **`.nextStep`**: Resets on the following step call. When autoreset occurs, returns the new observation with `reward=0`, `terminated=false`, `truncated=false` (the action is ignored for that step).

```swift
import Gymnazo

var env: any Env = CartPole().autoReset(mode: .sameStep)
```

## Topics

### Core Protocol

- ``Wrapper``

### Time and Episode Management

- ``TimeLimit``
- ``RecordEpisodeStatistics``
- ``AutoReset``

### Validation

- ``OrderEnforcing``
- ``PassiveEnvChecker``

### Observation Wrappers

- ``TransformObservation``
- ``NormalizeObservation``
- ``FlattenObservation``

### Reward Wrappers

- ``TransformReward``
- ``NormalizeReward``

### Action Wrappers

- ``ClipAction``
- ``RescaleAction``
