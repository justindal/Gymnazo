# Gymnazo

[![Swift](https://github.com/justindal/Gymnazo/actions/workflows/main.yml/badge.svg)](https://github.com/justindal/Gymnazo/actions/workflows/main.yml)

**Gymnazo** is a reinforcement learning toolkit written in Swift for Apple platforms. It provides a collection of Gymnasium-style environments and utilities for building and testing reinforcement learning algorithms, with implementations of common reinforcement learning algorithms, inspired by Stable-Baselines3. Gymnazo uses [MLX Swift](https://github.com/ml-explore/mlx-swift) to benefit Apple Silicon devices.

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
    .package(url: "https://github.com/justindal/Gymnazo.git", from: "0.1.0")
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
  test
```

Or open the package in Xcode and build/test.

## Quick Start

Gymnazo closely follows the Gymnasium API:

```swift
import Gymnazo
import MLX

@main
struct Demo {
    @MainActor
    static func main() {
        var env = Gymnazo.make("CartPole")
        let reset = env.reset(seed: 42)
        var observation = reset.obs
        var key = MLX.key(42)
        var done = false
        var totalReward = 0.0

        while !done {
            let action = env.action_space.sample(key: key)
            let step = env.step(action)
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
> Note: Vector environments are currently in development and may not be fully functional.

For parallel training, Gymnazo provides vector environments (sync + async):

```swift
import Gymnazo

// Synchronous vector environment with 4 CartPoles
let syncEnv = Gymnazo.make_vec("CartPole", numEnvs: 4)
let reset = syncEnv.reset(seed: 42)
_ = reset.observations

// Asynchronous vector environment with 4 CartPoles
let asyncEnv = Gymnazo.make_vec_async("CartPole", numEnvs: 4)
let step = await asyncEnv.stepAsync([1, 0, 1, 0])
_ = step.rewards
```

## Wrappers

Wrappers let you modify environment behavior without changing its implementation.

When you call `Gymnazo.make(...)`, wrappers are applied automatically in the following order:

1. `PassiveEnvChecker` (disable with `disableEnvChecker: true`)
2. `OrderEnforcing`
3. `TimeLimit` (controlled via `maxEpisodeSteps`, use `-1` to disable)

```swift
import Gymnazo

// default
var env = Gymnazo.make("CartPole")

// customize environment behavior
var customEnv = Gymnazo.make(
    "CartPole",
    maxEpisodeSteps: 500,
    disableEnvChecker: true,
    recordEpisodeStatistics: true
)

// wrappers can also be applied using the chainable extension methods
let chainedEnv = CartPole()
    .validated(maxSteps: 500)
    .observationsNormalized()
```

## Environment Catalog

Gymnazo currently contains the following environments:

| Category            | Environments                                                                 |
| ------------------- | ---------------------------------------------------------------------------- |
| **Classic Control** | `CartPole`, `MountainCar`, `MountainCarContinuous`, `Acrobot`, `Pendulum`    |
| **Box2D**           | `LunarLander`, `LunarLanderContinuous`, `CarRacing`, `CarRacingDiscrete`     |
| **ToyText**         | `FrozenLake`, `Taxi`, `CliffWalking`, `Blackjack`                            |

See the [documentation](https://swiftpackageindex.com/justindal/gymnazo/main/documentation/gymnazo/creating-environments#Registering-a-custom-environment) to learn more about the included environments and how to register custom environments.

## Spaces

Spaces describe valid actions/observations:

`Discrete`, `Box`, `MultiDiscrete`, `MultiBinary`, `TextSpace`, `Tuple`, `Dict`, `SequenceSpace`, `Graph`

## Included Reinforcement Learning Algorithms (Experimental)

Gymnazo includes common reinforcement learning algorithms implementations, inspired by Stable-Baselines3.

> This has not been tested much yet.

- `SAC` (Soft Actor-Critic)

```swift
import Gymnazo

let env = Gymnazo.make("Pendulum")
let model = SAC(env: env)
model.learn(totalTimesteps: 100000)
```

## Documentation
To learn more about Gymnazo, please refer to the 
[documentation](https://swiftpackageindex.com/justindal/Gymnazo/main/documentation/gymnazo) hosted on Swift Package Index.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Farama Foundation](https://farama.org/) for Gymnasium
- [OpenAI](https://openai.com/) for the original Gym
- [MLX Swift](https://github.com/ml-explore/mlx-swift)
- [Box2D](https://box2d.org/) for the physics engine used in some environments
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) as a guide for the RL implementations
