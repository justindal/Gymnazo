## ExploreRLCore

ExploreRLCore is a Swift package that ports core ideas from the Farama Gymnasium project to Swift. It provides Gymnasium-style environments, spaces, and wrappers on top of MLX Swift so you can train reinforcement learning agents on Apple platforms.

ExploreRLCore also powers my companion app, ExploreRL, available on macOS and iOS, which helps you explore reinforcement learning environments and algorithms interactively.

### Features

- **Gymnasium-style API**: `Environment` protocol, `Gymnasium.start()`, `Gymnasium.make()` and `env.step` / `env.reset` signatures that mirror Python Gymnasium.
- **Spaces**: `Discrete` and `Box` spaces, with more planned (`MultiDiscrete`, `MultiBinary`, `Tuple`, `Dict`).
- **Environments**:
  - `FrozenLake-v1` from the Toy Text family, including optional random map generation.
- **Wrappers**:
  - passive env checker, order enforcing, time limit, episode statistics.
  - action wrappers for `Box` spaces (`ClipAction`, `RescaleAction`).
- **RL utilities**: a tabular `QLearningAgent` backed by MLX arrays.

### Installation

Add the package to your `Package.swift` dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/justindal/explorerlcore.git", from: "0.1.0")
]
```

Then add `ExploreRLCore` to your target dependencies:

```swift
.target(
    name: "myapp",
    dependencies: [
        .product(name: "ExploreRLCore", package: "ExploreRLCore")
    ]
)
```

### Basic usage

```swift
import ExploreRLCore
import MLXRandom

@main
struct Demo {
    static func main() {
        Gymnasium.start()

        var env = Gymnasium.make("FrozenLake-v1",
                                 kwargs: ["is_slippery": true])

        var (obs, _) = env.reset(seed: 42)
        var done = false

        while !done {
            let action = env.action_space.sample(key: MLXRandom.key(0))
            let step = env.step(action)
            obs = step.obs
            done = step.terminated || step.truncated
            print("state: \(obs), reward: \(step.reward)")
        }

        env.render()
    }
}
```

### Example training loop

The repository includes an example training loop you can adapt. It might look something like this:

```swift
Gymnasium.start()

var env = Gymnasium.make("FrozenLake-v1",
                         kwargs: ["is_slippery": true]) as! FrozenLake

let nStates = env.observation_space.n
let nActions = env.action_space.n

var agent = QLearningAgent(
    learningRate: 0.1,
    gamma: 0.99,
    stateSize: nStates,
    actionSize: nActions,
    epsilon: 0.1
)

var key = MLXRandom.key(0)

for episode in 0..<1000 {
    var (obs, _) = env.reset(seed: UInt64(episode))
    var done = false

    while !done {
        let action = agent.chooseAction(
            actionSpace: env.action_space,
            state: obs,
            key: &key
        )

        let step = env.step(action)
        let nextObs = step.obs
        let reward = Float(step.reward)

        _ = agent.update(
            state: obs,
            action: action,
            reward: reward,
            nextState: nextObs
        )

        obs = nextObs
        done = step.terminated || step.truncated
    }
}
```

### SwiftUI rendering

For SwiftUI, you can embed the FrozenLake canvas directly using the snapshot API:

```swift
import SwiftUI
import ExploreRLCore

@MainActor
struct FrozenLakeView: View {
    @State private var env = Gymnasium.make(
        "FrozenLake-v1",
        kwargs: ["is_slippery": true]
    ) as! FrozenLake

    var body: some View {
        FrozenLakeCanvasView(snapshot: env.currentSnapshot)
            .frame(maxWidth: 400, maxHeight: 400)
    }
}
```

### Testing and building

Because ExploreRLCore depends on MLX Swift, which uses Metal shaders, the package cannot currently be built or tested reliably with the pure SwiftPM command-line tools (`swift build`, `swift test`). Instead, use Xcode or `xcodebuild`:

- open the package in Xcode and run the `ExploreRLCore-Package` test scheme, or
- from the command line:

  ```bash
  xcodebuild \
    -scheme ExploreRLCore-Package \
    -destination 'platform=macOS,arch=arm64' \
    test
  ```

This exercises spaces, wrappers, and the `FrozenLake` environment using Xcode's build system.

### Status and roadmap

This package currently includes a subset of Gymnasium functionality focused on Toy Text and basic spaces. Planned work includes:

- additional spaces: `MultiDiscrete`, `MultiBinary`, `Tuple`, `Dict`.
- more wrappers: observation and reward normalization, frame stacking.
- vectorized environments.
- more reference environments (CartPole, MountainCar, Taxi).

Contributions and feedback are welcome.
