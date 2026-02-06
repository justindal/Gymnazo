# Core Concepts

The key building blocks in Gymnazo are **environments**, **spaces**, and **wrappers**.

## Environments

An environment conforms to ``Env`` and defines:

- An `actionSpace` and `observationSpace` (both `any Space`)
- `reset(seed:options:)` → ``Reset`` (initial observation + info)
- `step(_:)` → ``Step`` (next observation + reward + termination flags + info)

All observations and actions are `MLXArray`.

```swift
import Gymnazo
import MLX

var env = try await Gymnazo.make("CartPole")

let reset = try env.reset(seed: 42, options: nil)
var done = false
var key = MLX.key(0)

while !done {
    let (newKey, sampleKey) = MLX.split(key: key)
    key = newKey
    let action = env.actionSpace.sample(key: sampleKey)
    let step = try env.step(action)
    done = step.terminated || step.truncated
}
```

## Spaces

Spaces describe the set of valid actions/observations. All space operations use `MLXArray`. See <doc:Spaces>.

## Wrappers

Wrappers help to change the behavior of an environment (validation, time limits, normalization, autoreset, etc.) without modifying the underlying environment. See <doc:Wrappers-Gym>.

## Topics

- ``Env``
- ``Step``
- ``Reset``
- ``Space``
- ``Wrapper``
