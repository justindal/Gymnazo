# LunarLander

A classic rocket trajectory optimization problem from the Box2D suite.

## Overview

Land a rocket on the landing pad smoothly. This is the **discrete-action** variant. For continuous actions, see <doc:LunarLanderContinuous>.

## Action Space

Four discrete actions:

| Action | Description                    |
|--------|--------------------------------|
| 0      | Do nothing                     |
| 1      | Fire left orientation engine   |
| 2      | Fire main engine               |
| 3      | Fire right orientation engine  |

## Observation Space

An `MLXArray` of shape `(8,)`:

| Index | Observation               |
|------:|---------------------------|
| 0     | X position                |
| 1     | Y position                |
| 2     | X velocity                |
| 3     | Y velocity                |
| 4     | Angle                     |
| 5     | Angular velocity          |
| 6     | Left leg contact (0/1)    |
| 7     | Right leg contact (0/1)   |

## Rewards

Shaped reward encouraging:

- proximity to landing pad
- low velocity and low tilt
- leg contacts

Plus a terminal bonus/penalty for safe landing vs crash. Engine usage incurs small per-frame penalties.

## Episode Termination

Terminates when the lander crashes, goes out of bounds, or lands successfully (stable contact conditions).

Truncation is typically handled by the default `maxEpisodeSteps` wrapper for the registered env.

## Arguments

```swift
import Gymnazo

let env = Gymnazo.make("LunarLander", kwargs: [
    "render_mode": "rgb_array",
    "gravity": -10.0,
    "enable_wind": false,
    "wind_power": 15.0,
    "turbulence_power": 1.5
])
```

- `render_mode` (String, optional): `"human"` or `"rgb_array"`.
- `gravity` (Float, default `-10.0`): gravitational constant used by the physics.
- `enable_wind` (Bool, default `false`): enables wind/turbulence effects.
- `wind_power` (Float, default `15.0`): wind magnitude.
- `turbulence_power` (Float, default `1.5`): turbulence magnitude.

## Topics

### Supporting Types

- ``LunarLanderSnapshot``


