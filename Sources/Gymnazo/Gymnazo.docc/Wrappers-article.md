# Wrappers

Modify environment behavior using composable wrappers.

## Overview

Wrappers allow you to modify an environment's behavior without changing its underlying implementation. They follow the decorator pattern—each wrapper wraps an environment (or another wrapper) and intercepts calls to `step(_:)`, `reset(seed:options:)`, and other methods.

## Default Wrappers via make()

When you call ``make(_:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:kwargs:)-(String,_,_,_,_,_,_,_)``, wrappers are applied automatically in this order:

1. **PassiveEnvChecker** - Validates API compliance (disable with `disableEnvChecker: true`)
2. **OrderEnforcing** - Ensures `reset(seed:options:)` is called before `step(_:)`
3. **TimeLimit** - Truncates episodes at `maxEpisodeSteps` (if the environment defines one)

Note: `RecordEpisodeStatistics` is **not** applied by default. Enable it explicitly if needed.

```swift
import Gymnazo

// Create environment with default wrappers
var env = Gymnazo.make("CartPole-v1")

// Customize wrapper behavior
var env = Gymnazo.make(
    "CartPole-v1",
    maxEpisodeSteps: 500,           // Override default time limit
    disableEnvChecker: true,        // Disable PassiveEnvChecker
    recordEpisodeStatistics: true   // Enable RecordEpisodeStatistics
)

// Use maxEpisodeSteps: -1 to disable TimeLimit entirely
var env = Gymnazo.make("CartPole-v1", maxEpisodeSteps: -1)
```

## Chainable Wrapper Extensions

Gymnazo provides chainable extension methods for applying wrappers beyond what ``make(_:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:kwargs:)-(String,_,_,_,_,_,_,_)`` provides:

```swift
import Gymnazo

// Start with make() for default wrappers, then add extras
var env = Gymnazo.make("MountainCarContinuous-v0")
    .actionsRescaled(from: (-1.0, 1.0))  // Rescale agent outputs
    .observationsNormalized()             // Normalize observations

// Or for custom setups without make()
let env = CartPole()
    .validated(maxSteps: 500)       // Standard wrapper stack
    .observationsTransformed { $0 / 10.0 }  // Custom normalization
```

### Wrapper Order

Wrappers are applied inside-out—the **last wrapper in the chain processes calls first**.

```swift
// Example: Adding observation normalization to an environment
var env = Gymnazo.make("CartPole-v1")
    .observationsNormalized()  // Applied after make()'s wrappers
```

Call flow: `Your code → NormalizeObservation → [make() wrappers] → CartPole`

> **Tip:** Use `.validated(maxSteps:)` when not using `make()` to apply the standard stack (PassiveEnvChecker + OrderEnforcing + TimeLimit):
>
> ```swift
> let env = CartPole()
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

var env = CartPole()
let wrapped = TimeLimit(env: env, maxEpisodeSteps: 500)
```

## Available Wrappers

### TimeLimit

Truncates episodes after a maximum number of steps:

```swift
var env = TimeLimit(env: CartPole(), maxEpisodeSteps: 200)

// After 200 steps, truncated will be true
let (_, _, _, truncated, _) = env.step(action)
```

### RecordEpisodeStatistics

Tracks episode returns, lengths, and timing. **Not applied by default** - enable with `recordEpisodeStatistics: true`.

```swift
// Enable via make()
var env = Gymnazo.make("CartPole-v1", recordEpisodeStatistics: true)

// Or apply manually
var env = RecordEpisodeStatistics(env: CartPole(), bufferLength: 100)

// Run some episodes...

// Access statistics (stored in queues)
env.returnQueue   // Last 100 episode returns
env.lengthQueue   // Last 100 episode lengths
env.timeQueue     // Last 100 episode durations
env.episodeCount  // Total episodes completed
```

When an episode ends, the statistics are also added to the step's info dictionary under the `"episode"` key:

```swift
let result = env.step(action)
if result.terminated || result.truncated {
    if let stats = result.info["episode"] as? [String: Any] {
        let episodeReturn = stats["r"] as! Double
        let episodeLength = stats["l"] as! Int
        let episodeTime = stats["t"] as! Double
    }
}
```

### OrderEnforcing

Ensures `reset(seed:options:)` is called before `step(_:)`. Applied by default via `make(_:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:kwargs:)-(String,_,_,_,_,_,_,_)`.

```swift
var env = OrderEnforcing(env: CartPole())

// This will trigger an assertion failure:
// env.step(0)  // Error: reset() not called

let _ = env.reset()
let _ = env.step(action)  // OK
```

### PassiveEnvChecker

Validates environment API compliance during runtime. Applied by default via `make(_:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:kwargs:)-(String,_,_,_,_,_,_,_)`.

```swift
var env = PassiveEnvChecker(env: CartPole())

// Automatically checks:
// - Observation matches observation_space
// - Action is valid for action_space
// - Return types are correct
```

Disable with `Gymnazo.make("CartPole-v1", disableEnvChecker: true)`.

### TransformObservation

Applies a custom transformation to observations:

```swift
var env = TransformObservation(
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
var env = NormalizeObservation(env: CartPole())

// Observations are automatically normalized to approximately N(0, 1)
```

### ClipAction

Clips continuous actions to the action space bounds:

```swift
var env = ClipAction(env: MountainCarContinuous())

// Actions outside [-1, 1] are automatically clipped
let clippedStep = env.step(MLXArray([2.0]))  // Clipped to 1.0
```

### RescaleAction

Rescales actions from a source range (default `[-1, 1]`) to the environment's action space bounds:

```swift
var env = RescaleAction(
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

var env = CartPole()
let checked = PassiveEnvChecker(env: env)
let limited = TimeLimit(env: checked, maxEpisodeSteps: 500)
let recorded = RecordEpisodeStatistics(env: limited)
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

Create your own wrapper by conforming to the `Wrapper` protocol. The protocol is generic over the inner environment type:

```swift
import MLX

public final class RewardScaler<InnerEnv: Env>: Wrapper {
    public typealias Observation = InnerEnv.Observation
    public typealias Action = InnerEnv.Action
    public typealias ObservationSpace = InnerEnv.ObservationSpace
    public typealias ActionSpace = InnerEnv.ActionSpace

    public var env: InnerEnv
    public let scale: Double

    public required init(env: InnerEnv) {
        self.env = env
        self.scale = 1.0
    }

    public init(env: InnerEnv, scale: Double) {
        self.env = env
        self.scale = scale
    }

    public func step(_ action: Action) -> StepResult {
        let result = env.step(action)
        return (
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
var env = RewardScaler(env: CartPole(), scale: 0.01)
```

### FlattenObservation

Flattens observations into a 1D `MLXArray` and updates the observation space to a flat `Box`. This wrapper only works when `flatten_space(env.observation_space)` returns a `Box`. For `SequenceSpace` and `Graph`, `flatten_space` preserves the structured space instead of forcing a single `Box`.

```swift
import Gymnazo

var env = FrozenLake().observationsFlattened()
let (obs, _) = env.reset(seed: 0)
```

### TransformReward

Applies a custom function to each reward.

```swift
import Gymnazo

var env = CartPole().rewardsTransformed { $0 * 2 }
```

### NormalizeReward

Normalizes rewards using a running return variance estimate.

```swift
import Gymnazo

var env = CartPole().rewardsNormalized()
```

### AutoReset

Automatically resets when an episode ends. In `sameStep` mode, the observation returned is from the new episode and the previous episode’s final observation is stored in `info["final_observation"]`.

```swift
import Gymnazo

var env = CartPole().autoReset(mode: .sameStep)
```

## Topics

### Core Protocol

- `Wrapper`

### Time and Episode Management

- `TimeLimit`
- `RecordEpisodeStatistics`
- `AutoReset`

### Validation

- `OrderEnforcing`
- `PassiveEnvChecker`

### Observation Wrappers

- `TransformObservation`
- `NormalizeObservation`
- `FlattenObservation`

### Reward Wrappers

- `TransformReward`
- `NormalizeReward`

### Action Wrappers

- `ClipAction`
- `RescaleAction`
