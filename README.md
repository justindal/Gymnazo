# Gymnazo

[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fjustindal%2FGymnazo%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/justindal/Gymnazo)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fjustindal%2FGymnazo%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/justindal/Gymnazo)
[![Swift Tests](https://github.com/justindal/Gymnazo/actions/workflows/main.yml/badge.svg)](https://github.com/justindal/Gymnazo/actions/workflows/main.yml)

Gymnazo is a reinforcement learning toolkit written in Swift for Apple platforms. It provides Gymnasium-style environments, vector collection, and utilities for building and testing reinforcement learning algorithms, with implementations inspired by Stable-Baselines3. Gymnazo uses [MLX Swift](https://github.com/ml-explore/mlx-swift) to benefit Apple Silicon devices.

## Requirements

- macOS 15+ / iOS 18+
- Apple Silicon (M-series or A-series chips)
- Swift 6.0+

## Installation

### Xcode

Add `https://github.com/justindal/Gymnazo.git` as a package dependency and add Gymnazo as a dependency to your target.

### Swift Package Manager

Add Gymnazo to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/justindal/Gymnazo.git", from: "0.12.0")
]
```

Then add it to your target:

```swift
.target(
    name: "App",
    dependencies: [
        .product(name: "Gymnazo", package: "Gymnazo")
    ]
)
```
## Building & Testing

> **Note**: Gymnazo might not fully build with the Swift Package Manager command line (`swift build`) as SPM cannot build Metal shaders. Build with Xcode or `xcodebuild`.

```bash
xcodebuild \
  -scheme Gymnazo-Package \
  -destination 'platform=macOS,arch=arm64' \
  -parallel-testing-enabled NO \
  test
```

Or open the package in Xcode and build/test.

The package also includes an Apple Silicon profiler for typed collection, batched replay insertion, single and vector DQN training, PPO, CNN training, memory usage, and compile warm-up:

```bash
xcodebuild \
  -scheme GymnazoProfiler \
  -destination 'platform=macOS,arch=arm64' \
  -configuration Release \
  -derivedDataPath /tmp/gymnazo-profiler \
  ENABLE_TESTABILITY=NO ENABLE_CODE_COVERAGE=NO \
  build
/tmp/gymnazo-profiler/Build/Products/Release/GymnazoProfiler
```

## Quick Start

Gymnazo closely follows the Gymnasium API:

```swift
import Gymnazo
import MLX

@main
struct Demo {
    @MainActor
    static func main() async throws {
        var env = try await Gymnazo.make("CartPole")
        let reset = try env.reset()
        var observation = reset.obs
        var key = MLX.key(42)
        var done = false
        var totalReward = 0.0

        while !done {
            let (newKey, actionKey) = MLX.split(key: key)
            key = newKey
            let action = env.actionSpace.sample(key: actionKey)

            let step = try env.step(action)

            totalReward += step.reward
            observation = step.obs
            done = step.terminated || step.truncated
        }

        _ = observation
        print("Episode finished with reward: \(totalReward)")
    }
}
```

## Vector Environments

Gymnazo provides compatibility vector environments plus a typed Swift-value vector boundary. Use the typed boundary when environment state should stay out of MLX and concurrency transfers:

```swift
import Gymnazo

let vector = try await Gymnazo.makeValueVec(
    "CartPole",
    numEnvs: 4,
    mode: .parallel,
    autoresetMode: .sameStep
)
let reset = try await vector.reset(seed: 42)
let step = try await vector.step([
    .discrete(0), .discrete(1), .discrete(0), .discrete(1)
])
_ = (reset.observations, step.rewards)
try await vector.close()
```

The compatibility vector APIs remain available for existing `Env`/`MLXArray` integrations:

```swift
import MLX

let syncEnv = try await Gymnazo.makeVec("CartPole", numEnvs: 4)
let actions = MLXArray([1, 0, 1, 0])
let step = try syncEnv.step(actions)
_ = step.rewards
```

`AsyncVectorEnv` preserves the compatibility API but serializes access through the main actor. Use `makeValueVec(..., mode: .parallel)` for actor-owned parallel collection.

## Wrappers

Wrappers let you modify environment behavior without changing its implementation.

When you call `Gymnazo.make(...)`, wrappers are applied automatically in the following order:

1. `PassiveEnvChecker` (disable with `disableEnvChecker: true`)
2. `OrderEnforcing`
3. `TimeLimit` (controlled via `maxEpisodeSteps`, use `-1` to disable)

```swift
import Gymnazo

// default
var env = try await Gymnazo.make("CartPole")

// customize environment behavior
var customEnv = try await Gymnazo.make(
    "CartPole",
    maxEpisodeSteps: 500,
    disableEnvChecker: true,
    recordEpisodeStatistics: true
)

// wrappers can also be applied using the chainable extension methods
let chainedEnv = try CartPole()
    .validated(maxSteps: 500)
    .observationsNormalized()
```

Typed value environments provide the same composition model without `MLXArray` crossing environment boundaries. Available typed wrappers include time limits, autoreset, episode statistics, observation/action/reward transforms, reshape, delay, time-aware observations, max-and-skip, sticky actions, reward clipping, and dictionary/tuple filtering.

## Environment Catalog

Gymnazo currently contains the following environments:

| Category            | Environments                                                                 |
| ------------------- | ---------------------------------------------------------------------------- |
| **Classic Control** | `CartPole`, `MountainCar`, `MountainCarContinuous`, `Acrobot`, `Pendulum`    |
| **Arcade**          | `FlappyBird`, `Snake`, `2048`                                                 |
| **Box2D**           | `LunarLander`, `LunarLanderContinuous`, `CarRacing`, `CarRacingDiscrete`     |
| **ToyText**         | `FrozenLake`, `Taxi`, `CliffWalking`, `Blackjack`                            |

See the [documentation](https://swiftpackageindex.com/justindal/gymnazo/main/documentation/gymnazo/creating-environments#Registering-a-custom-environment) to learn more about the included environments and how to register custom environments.

## Spaces

Spaces describe valid actions/observations:

`Discrete`, `Box`, `MultiDiscrete`, `MultiBinary`, `TextSpace`, `Tuple`, `Dict`, `SequenceSpace`, `Graph`

## Reinforcement Learning Algorithms

Gymnazo includes common reinforcement learning algorithm implementations inspired by Stable-Baselines3. Algorithms have support for callbacks, checkpointing, and save/load.

| Algorithm | Type | Action Space |
|-----------|------|--------------|
| `DQN` | Deep Q-Network | Discrete |
| `SAC` | Soft Actor-Critic | Continuous |
| `PPO` | Proximal Policy Optimization | Discrete + Continuous + MultiDiscrete + MultiBinary |
| `TD3` | Twin Delayed DDPG | Continuous |
| `TabularAgent` | Q-Learning / SARSA | Discrete (small) |

```swift
import Gymnazo

let env = try await Gymnazo.make("CartPole")
let model = try DQN(env: env)
try await model.learn(totalTimesteps: 50_000, callbacks: nil)

let ppo = try PPO(
    env: env,
    config: PPOConfig(nSteps: 128, batchSize: 32, nEpochs: 4)
)
try await ppo.learn(totalTimesteps: 50_000, callbacks: nil)

// save and load checkpoints
let url = URL.documentsDirectory.appending(path: "my-agent")
try await model.save(to: url)
let loaded = try DQN.load(from: url, env: env)
```

See the [Reinforcement Learning guide](https://swiftpackageindex.com/justindal/Gymnazo/main/documentation/gymnazo/reinforcement-learning) for detailed usage, configuration, and callbacks.

For typed policy inference and collection, see the [typed value boundary guide](https://swiftpackageindex.com/justindal/Gymnazo/main/documentation/gymnazo/typed-value-boundary).

## TODO
- [ ] add end-to-end training lifecycle coverage for all bundled algorithms
- [ ] move ExploreRL training collection to typed vectors and typed replay/rollout ingestion
- [ ] expand exact seeded traces to every bundled environment
- [ ] add optional environment families after typed boundary validation



## Documentation
To learn more about Gymnazo, please refer to the 
[documentation](https://swiftpackageindex.com/justindal/Gymnazo/main/documentation/gymnazo) hosted on Swift Package Index.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Farama Foundation](https://farama.org/)
- [OpenAI](https://openai.com/)
- [MLX Swift](https://github.com/ml-explore/mlx-swift)
- [Box2D](https://box2d.org/)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
