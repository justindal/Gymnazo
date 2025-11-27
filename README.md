# Gymnazo

[![Swift](https://github.com/justindal/Gymnazo/actions/workflows/main.yml/badge.svg)](https://github.com/justindal/Gymnazo/actions/workflows/main.yml)

**Gymnazo** is a Swift port of [Farama Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for reinforcement learning. Built on [MLX Swift](https://github.com/ml-explore/mlx-swift) for Apple Silicon acceleration. Under active development.

## Requirements

- **macOS 14+** / **iOS 17+**
- **Apple Silicon** (M-series or A-series chips)
- **Swift 6.0+**

> **Note**: Gymnazo uses MLX Swift which requires Metal shaders, and will not build with SwiftPM (`swift build`). Build with Xcode or `xcodebuild`.

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
| **Classic Control** | `CartPole-v1`, `MountainCar-v0`, `MountainCarContinuous-v0`, `Acrobot-v1` |
| **Toy Text**        | `FrozenLake-v1`, `FrozenLake8x8-v1`                                       |

### Spaces

`Discrete`, `Box`, `MultiDiscrete`, `Tuple`, `Dict`

### Wrappers

`TimeLimit`, `RecordEpisodeStatistics`, `ClipAction`, `RescaleAction`, `TransformObservation`, `NormalizeObservation`, `OrderEnforcing`, `PassiveEnvChecker`

### Built-in RL Algorithms

| Algorithm        | Type    | Description                              |
| ---------------- | ------- | ---------------------------------------- |
| `QLearningAgent` | Tabular | Q-learning with ε-greedy exploration     |
| `SARSAAgent`     | Tabular | On-policy TD control                     |
| `DQNAgent`       | Deep RL | Deep Q-Network with replay buffer        |
| `SACAgent`       | Deep RL | Soft Actor-Critic for continuous actions |

## Quick Start

```swift
import Gymnazo
import MLX

@main
struct Demo {
    @MainActor
    static func main() {
        Gymnazo.start()

        var env = Gymnazo.make("CartPole-v1")
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
- 6 environments (Classic Control + Toy Text)
- 8 wrappers
- 4 RL algorithms (Q-Learning, SARSA, DQN, SAC)

**Planned:**

- All environments from the original Gymnasium
- Vectorized environments
- Additional RL algorithms
- Better wrapper support

For the full Gymnasium experience, see the [official Python repository](https://github.com/Farama-Foundation/Gymnasium).

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Farama Foundation](https://farama.org/) for the original Gymnasium
- [OpenAI](https://openai.com/) for the original Gym
- [MLX Swift](https://github.com/ml-explore/mlx-swift) team
