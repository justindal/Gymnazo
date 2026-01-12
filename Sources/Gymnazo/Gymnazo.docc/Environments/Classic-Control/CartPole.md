# CartPole

A classic cart-and-pole balancing task from the Classic Control suite.

## Overview

Balance a pole attached by an un-actuated joint to a cart moving along a frictionless track.

## Action Space

Two discrete actions:

| Action | Meaning     |
|--------|-------------|
| 0      | Push left   |
| 1      | Push right  |

## Observation Space

An `MLXArray` of shape `(4,)` containing:

| Index | Observation          |
|------:|----------------------|
| 0     | Cart position `x`    |
| 1     | Cart velocity `ẋ`   |
| 2     | Pole angle `θ`       |
| 3     | Pole angular velocity `θ̇` |

## Rewards

`+1.0` for each step (including the termination step).

## Episode Termination

Terminates when either:

- The cart position exceeds the position threshold.
- The pole angle exceeds the angle threshold.

Truncation is typically handled by the default `maxEpisodeSteps` wrapper for the registered env.

## Arguments

```swift
import Gymnazo

let env = Gymnazo.make("CartPole", kwargs: [
    "render_mode": "human"
])
```

- `render_mode` (String, optional): `"human"` or `"rgb_array"`.

Note: `"rgb_array"` is currently not implemented for this environment and returns `nil`.

## Topics

### Supporting Types

- ``CartPoleSnapshot``


