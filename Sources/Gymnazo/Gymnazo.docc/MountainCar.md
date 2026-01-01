# MountainCar

A classic control task where the agent must drive up a hill by building momentum.

## Overview

The car’s engine is not powerful enough to climb the hill directly. The agent must first drive left to build up speed, then drive right to reach the goal.

## Action Space

Three discrete actions:

| Action | Meaning    |
| ------ | ---------- |
| 0      | Push left  |
| 1      | No push    |
| 2      | Push right |

## Observation Space

An `MLXArray` of shape `(2,)`:

| Index | Observation |
| ----: | ----------- |
|     0 | Position    |
|     1 | Velocity    |

## Rewards

`-1` for each step.

## Episode Termination

Terminates when:

- `position >= 0.5` and `velocity >= goal_velocity`

Truncation is typically handled by the default `maxEpisodeSteps` wrapper for the registered env.

## Arguments

```swift
let env = make("MountainCar", kwargs: [
    "render_mode": "human",
    "goal_velocity": 0.0
])
```

- `render_mode` (String, optional): `"human"` or `"rgb_array"`.
- `goal_velocity` (Float, default 0.0): required velocity at the goal position to count as success.

Note: `"rgb_array"` is currently not implemented for this environment and returns `nil`.

## Topics

### Environment Types

- `MountainCar`

### Supporting Types

- `MountainCarSnapshot`
