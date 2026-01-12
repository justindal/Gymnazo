# Creating Environments

Gymnazo uses a registry (similar to Gymnasium) so you can create environments by string ID.

## Creating by ID

```swift
import Gymnazo

var env = Gymnazo.make("CartPole")
```

## Keyword arguments (`kwargs`)

Some environments accept additional keyword arguments (for example, render mode or physics parameters):

```swift
import Gymnazo

let env = Gymnazo.make("LunarLander", kwargs: [
    "render_mode": "rgb_array",
    "enable_wind": false
])
```

`kwargs` is a `[String: Any]` dictionary, and each environment validates and interprets supported keys.

## Default wrappers applied by `make`

By default, `Gymnazo.make(...)` applies wrappers like ``PassiveEnvChecker``, ``OrderEnforcing``, and ``TimeLimit``. You can control that behavior via parameters like `disableEnvChecker`, `disableRenderOrderEnforcing`, and `maxEpisodeSteps`.

## Registering a custom environment

You can register your own environment under an ID and then create it with `Gymnazo.make(...)`:

```swift
import Gymnazo
import MLX

struct MyEnv: Env {
    typealias Observation = MLXArray
    typealias Action = Int
    typealias ObservationSpace = Box
    typealias ActionSpace = Discrete

    var action_space: Discrete { Discrete(2) }
    var observation_space: Box { Box(low: -1, high: 1, shape: [4]) }
    var spec: EnvSpec?
    var render_mode: String?

    mutating func step(_ action: Int) -> Step<MLXArray> {
        Step(obs: MLXArray.zeros([4]), reward: 0, terminated: false, truncated: false)
    }

    mutating func reset(seed: UInt64?, options: [String : Any]?) -> Reset<MLXArray> {
        Reset(obs: MLXArray.zeros([4]))
    }
}

register(id: "MyEnv") { _ in
    MyEnv()
}

let env = Gymnazo.make("MyEnv")
```

## Topics

### Creating and Registering

- ``make(_:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:kwargs:)-(String,_,_,_,_,_,_,_)``
- ``make(_:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:kwargs:)-(EnvSpec,_,_,_,_,_,_,_)``
- ``register(id:entryPoint:maxEpisodeSteps:rewardThreshold:nondeterministic:)``

### Vector Creation

- ``make_vec(_:numEnvs:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:autoresetMode:kwargs:)``
- ``make_vec_async(_:numEnvs:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:autoresetMode:kwargs:)``
