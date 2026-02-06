# Spaces

Spaces describe the set of valid actions and observations for an environment.

Every ``Env`` has:

- `env.actionSpace` (actions your agent can take)
- `env.observationSpace` (observations the environment can emit)

Both are `any Space`, and all space operations use `MLXArray`.

## Sampling and validation

All spaces conform to ``Space``:

```swift
import Gymnazo
import MLX

let space = Discrete(3)      // {0, 1, 2}
let action = space.sample(key: MLX.key(0))  // Returns MLXArray (scalar)
let ok = space.contains(action)

// For discrete spaces, extract the integer value when needed:
let actionInt = action.item(Int.self)  // 0, 1, or 2
```

## Common space types

- ``Discrete``: discrete actions (sampled as scalar `MLXArray`)
- ``Box``: continuous tensors with per-dimension bounds
- ``MultiDiscrete`` / ``MultiBinary``: multi-dimensional discrete/binary spaces
- ``Tuple`` and ``Dict``: structured spaces (product spaces)
- ``TextSpace``: text samples
- ``SequenceSpace``: variable-length sequences
- ``Graph``: graph-structured samples

## Flattening

Use ``flatten_space(_:)`` to transform many structured spaces into a flat ``Box`` (when possible). ``SequenceSpace`` and ``Graph`` preserve structure and flatten their internal element/node/edge spaces.

## Topics

- ``Space``
- ``Discrete``
- ``Box``
- ``MultiDiscrete``
- ``MultiBinary``
- ``TextSpace``
- ``Tuple``
- ``Dict``
- ``SequenceSpace``
- ``Graph``
- ``flatten_space(_:)``
