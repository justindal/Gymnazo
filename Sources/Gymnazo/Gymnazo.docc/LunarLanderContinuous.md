# LunarLanderContinuous

A continuous-action variant of LunarLander (Box2D).

## Overview

Control a rocket with a 2D continuous action. For discrete actions, see <doc:LunarLander>.

## Action Space

Two continuous actions:

| Index | Action          | Range |
|------:|-----------------|-------|
| 0     | Main throttle   | -1 to +1 |
| 1     | Lateral control | -1 to +1 |

Semantics:

- Main engine: values ≤ 0 turn off; values > 0 fire at 50%–100% power.
- Lateral engines: \(|value| \le 0.5\) turns off; \(|value| > 0.5\) fires with proportional power.

## Observation Space

Same as ``LunarLander``: an `MLXArray` of shape `(8,)` containing position, velocity, angle, angular velocity, and leg contacts.

## Rewards

Shaped reward encouraging stable landings, with per-frame penalties for engine usage (main and side thrusters).

## Episode Termination

Terminates on crash, out-of-bounds, or successful landing.

Truncation is typically handled by the default `maxEpisodeSteps` wrapper for the registered env.

## Arguments

```swift
let env = make("LunarLanderContinuous", kwargs: [
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

### Environment Types

- ``LunarLanderContinuous``

### Supporting Types

- ``LunarLanderSnapshot``


