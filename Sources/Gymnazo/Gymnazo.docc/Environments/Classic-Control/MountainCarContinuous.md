# MountainCarContinuous

A continuous-control variant of MountainCar.

## Overview

Instead of choosing from three discrete pushes, the agent outputs a continuous force in \([-1, 1]\).

## Action Space

One continuous action (force):

- **Shape**: `(1,)`
- **Range**: \([-1, 1]\)

## Observation Space

An `MLXArray` of shape `(2,)`:

| Index | Observation |
|------:|-------------|
| 0     | Position |
| 1     | Velocity |

## Rewards

- `+100` on reaching the goal state
- Minus an action cost proportional to `force²`

## Episode Termination

Terminates when:

- `position >= 0.45` and `velocity >= goal_velocity`

Truncation is typically handled by the default `maxEpisodeSteps` wrapper for the registered env.

## Arguments

```swift
import Gymnazo

let env = try await Gymnazo.make("MountainCarContinuous", options: [
    "render_mode": "human",
    "goal_velocity": 0.0
])
```

- `render_mode` (String, optional): `"human"` or `"rgb_array"`.
- `goal_velocity` (Float, default 0.0): required velocity at the goal position to count as success.

Note: `"rgb_array"` is currently not implemented for this environment and returns `nil`.

 
