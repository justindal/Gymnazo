# Spaces

Spaces describe the set of valid actions and observations for an environment.

Every ``Env`` has:

- `env.action_space` (actions your agent can take)
- `env.observation_space` (observations the environment can emit)

## Sampling and validation

All spaces conform to ``Space``:

```swift
import Gymnazo
import MLX

let space = Discrete(3)      // {0, 1, 2}
let x = space.sample(key: MLX.key(0))
let ok = space.contains(x)
```

## Common space types

- ``Discrete``: integer actions in a range
- ``Box``: continuous tensors (backed by `MLXArray`) with per-dimension bounds
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
