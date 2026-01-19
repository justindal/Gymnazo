# Creating Environments

Gymnazo uses a registry (similar to Gymnasium) so you can create environments by string ID.

## Creating by ID

```swift
import Gymnazo

var env = try await Gymnazo.make("CartPole")
```

## Options

Some environments accept additional keyword arguments (for example, render mode or physics parameters):

```swift
import Gymnazo

let env = try await Gymnazo.make("LunarLander", options: [
    "render_mode": "rgb_array",
    "enable_wind": false
])
```

`options` is an `EnvOptions` dictionary, and each environment validates and interprets supported keys.

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
    var actionSpace: Discrete { Discrete(2) }
    var observationSpace: Box { Box(low: -1, high: 1, shape: [4]) }
    var spec: EnvSpec?
    var renderMode: RenderMode?

    mutating func step(_ action: Int) throws -> Step<MLXArray> {
        Step(obs: MLXArray.zeros([4]), reward: 0, terminated: false, truncated: false)
    }

    mutating func reset(seed: UInt64?, options: EnvOptions?) throws -> Reset<MLXArray> {
        Reset(obs: MLXArray.zeros([4]))
    }
}

await Gymnazo.register(id: "MyEnv", entryPoint: { _ in
    MyEnv()
})

let env = try await Gymnazo.make("MyEnv")
```

## Topics

### Creating and Registering

- ``Gymnazo/make(_:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:options:)-(String,_,_,_,_,_,_,_)``
- ``Gymnazo/make(_:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:options:)-(EnvSpec,_,_,_,_,_,_,_)``
- ``Gymnazo/register(id:entryPoint:maxEpisodeSteps:rewardThreshold:nondeterministic:)``

### Vector Creation

- ``Gymnazo/makeVec(_:numEnvs:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:autoresetMode:options:)``
- ``Gymnazo/makeVecAsync(_:numEnvs:maxEpisodeSteps:disableEnvChecker:disableRenderOrderEnforcing:recordEpisodeStatistics:recordBufferLength:recordStatsKey:autoresetMode:options:)``
