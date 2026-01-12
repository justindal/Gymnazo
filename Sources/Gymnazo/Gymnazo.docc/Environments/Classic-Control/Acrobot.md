# Acrobot

A two-link pendulum with only the second joint actuated (Classic Control).

## Overview

Swing the free end of a two-link chain above a target height by applying discrete torques at the actuated joint.

## Action Space

Three discrete actions:

| Action | Meaning |
|--------|---------|
| 0      | Apply -1 torque |
| 1      | Apply 0 torque  |
| 2      | Apply +1 torque |

## Observation Space

An `MLXArray` of shape `(6,)`:

| Index | Observation |
|------:|-------------|
| 0     | `cos(theta1)` |
| 1     | `sin(theta1)` |
| 2     | `cos(theta2)` |
| 3     | `sin(theta2)` |
| 4     | `dtheta1` |
| 5     | `dtheta2` |

## Rewards

`-1` each step until termination; `0` on the termination step.

## Episode Termination

Terminates when:

- \(-\cos(\theta_1) - \cos(\theta_2 + \theta_1) > 1.0\)

Truncation is typically handled by the default `maxEpisodeSteps` wrapper for the registered env.

## Arguments

```swift
import Gymnazo

let env = Gymnazo.make("Acrobot", kwargs: [
    "render_mode": "human",
    "torque_noise_max": 0.0
])
```

- `render_mode` (String, optional): `"human"` or `"rgb_array"`.
- `torque_noise_max` (Float, default 0.0): uniform noise added to the applied torque.

Note: `"rgb_array"` is currently not implemented for this environment and returns `nil`.

## Topics

### Supporting Types

- ``AcrobotSnapshot``


