# Pendulum

A classic continuous-control swing-up task (Classic Control).

## Overview

Apply continuous torque to keep an inverted pendulum upright. Unlike many tasks, `Pendulum` does **not** terminate early; it runs for the configured episode length.

## Action Space

One continuous action (torque):

- **Shape**: `(1,)`
- **Range**: \([-2, 2]\)

## Observation Space

An `MLXArray` of shape `(3,)`:

| Index | Observation |
|------:|-------------|
| 0     | `cos(theta)` |
| 1     | `sin(theta)` |
| 2     | `theta_dot` |

## Rewards

The reward is the negative of a quadratic cost on angle, angular velocity, and torque (see source for the exact coefficients).

## Episode Termination

`terminated` is always `false` for this environment. Truncation is typically handled by the default `maxEpisodeSteps` wrapper for the registered env.

## Arguments

```swift
let env = make("Pendulum", kwargs: [
    "render_mode": "human",
    "g": 10.0
])
```

- `render_mode` (String, optional): `"human"` or `"rgb_array"`.
- `g` (Float, default 10.0): gravitational constant used by the dynamics.

Reset supports optional bounds:

- `options["x_init"]` (Float, default π): initial angle sampled uniformly from `[-x_init, x_init]`
- `options["y_init"]` (Float, default 1.0): initial angular velocity sampled uniformly from `[-y_init, y_init]`

Note: `"rgb_array"` is currently not implemented for this environment and returns `nil`.

## Topics

### Environment Types

- ``Pendulum``

### Supporting Types

- ``PendulumSnapshot``


