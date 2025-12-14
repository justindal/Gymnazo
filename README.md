# Gymnazo

[![Swift](https://github.com/justindal/Gymnazo/actions/workflows/main.yml/badge.svg)](https://github.com/justindal/Gymnazo/actions/workflows/main.yml)

**Gymnazo** is a reinforcement learning library for Swift. Gymnazo provides a collection of environments and tools for developing and testing reinforcement learning algorithms. It is inspired by (and closely follows) [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) and is built with [MLX Swift](https://github.com/ml-explore/mlx-swift) for Apple Silicon. Under active development.

## Requirements

- **macOS 14+** / **iOS 17+**
- **Apple Silicon** (M-series or A-series chips)
- **Swift 6.0+**

> **Note**: Gymnazo uses MLX Swift which requires Metal shaders, and might not build with SwiftPM (`swift build`). Build with Xcode or `xcodebuild`.

## Installation

Add Gymnazo to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/justindal/Gymnazo.git", from: "0.1.0")
]
```

Then add it to your target:

```swift
.target(
    name: "YourApp",
    dependencies: [
        .product(name: "Gymnazo", package: "Gymnazo")
    ]
)
```

## Building & Testing

```bash
xcodebuild \
  -scheme Gymnazo \
  -destination 'platform=macOS,arch=arm64' \
  test
```

Or open the package in Xcode and run the test scheme.

## Features

### Environments

| Category            | Environments                                                              |
| ------------------- | ------------------------------------------------------------------------- |
| **Classic Control** | `CartPole`, `MountainCar`, `MountainCarContinuous`, `Acrobot`, `Pendulum` |
| **Box2D**           | `LunarLander`, `LunarLanderContinuous`                                    |
| **Toy Text**        | `FrozenLake`, `FrozenLake8x8`                                             |

### Spaces

`Discrete`, `Box`, `MultiDiscrete`, `MultiBinary`, `TextSpace`, `Tuple`, `Dict`, `SequenceSpace`, `Graph`

### Wrappers

`TimeLimit`, `RecordEpisodeStatistics`, `ClipAction`, `RescaleAction`, `TransformObservation`, `NormalizeObservation`, `TransformReward`, `NormalizeReward`, `AutoReset`, `FlattenObservation`, `OrderEnforcing`, `PassiveEnvChecker`

## Quick Start

```swift
import Gymnazo
import MLX

@main
struct Demo {
    @MainActor
    static func main() {
        var env = Gymnazo.make("CartPole")
        var (obs, _) = env.reset(seed: 42)
        var done = false

        while !done {
            let action = env.action_space.sample(key: MLX.key(0))
            let step = env.step(action)
            obs = step.obs
            done = step.terminated || step.truncated
        }
    }
}
```

## Status

Gymnazo is under active development.

**Implemented:**

- Core Gymnasium-style API (`Env`, `Space`, `Wrapper`)
- 9 environments (Classic Control, Box2D, Toy Text)
- 8 wrappers
- Vectorized environments (`SyncVectorEnv`, `AsyncVectorEnv`)
- SpriteKit/SwiftUI Canvas rendering for visualization

**Planned:**

- Additional environments from the original Gymnasium
- More wrapper types
- Intel Mac support

For the full Gymnasium experience, see the [official Python repository](https://github.com/Farama-Foundation/Gymnasium).

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Farama Foundation](https://farama.org/) for Gymnasium
- [OpenAI](https://openai.com/) for the original Gym
- [MLX Swift](https://github.com/ml-explore/mlx-swift)
- [Box2D](https://box2d.org/) for the physics engine used in some environments
