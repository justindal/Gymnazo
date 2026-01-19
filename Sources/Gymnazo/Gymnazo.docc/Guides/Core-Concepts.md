# Core Concepts

The key building blocks in Gymnazo are **environments**, **spaces**, and **wrappers**.

## Environments

An environment conforms to ``Env`` and defines:

- An `actionSpace` and `observationSpace`
- `reset(seed:options:)` → ``Reset`` (initial observation + info)
- `step(_:)` → ``Step`` (next observation + reward + termination flags + info)

```swift
import Gymnazo
import MLX

var env = try await Gymnazo.make("CartPole")

let reset = try env.reset(seed: 42, options: nil)
var done = false

while !done {
    let action = env.actionSpace.sample(key: MLX.key(0))
    let step = try env.step(action)
    done = step.terminated || step.truncated
}
```

## Spaces

Spaces describe the set of valid actions/observations (and how to sample them). See <doc:Spaces>.

## Wrappers

Wrappers help to change the behavior of an environment (validation, time limits, normalization, autoreset, etc.) without modifying the underlying environment. See <doc:Wrappers-Gym>.

## Topics

- ``Env``
- ``Step``
- ``Reset``
- ``Space``
- ``Wrapper``
