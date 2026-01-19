# FrozenLake

A classic grid-world navigation task (ToyText).

## Overview

Navigate a frozen lake from start to goal without falling into holes. The map can be deterministic (non-slippery) or stochastic (slippery).

Gymnazo registers two common IDs:

- `FrozenLake` (default map `"4x4"`)
- `FrozenLake8x8` (default map `"8x8"`)

## Action Space

Four discrete actions:

| Action | Meaning |
|--------|---------|
| 0      | Left    |
| 1      | Down    |
| 2      | Right   |
| 3      | Up      |

## Observation Space

The observation is an `Int` representing the agent's current state index in the grid.

## Rewards

Typical reward structure:

- `1.0` for reaching the goal
- `0.0` otherwise

## Episode Termination

Terminates when the agent reaches the goal or falls into a hole.

Truncation is typically handled by the default `maxEpisodeSteps` wrapper for the registered env.

## Arguments

```swift
import Gymnazo

let env = try await Gymnazo.make("FrozenLake", options: [
    "render_mode": "ansi",
    "map_name": "4x4",
    "is_slippery": true
])
```

- `render_mode` (String, optional): `"ansi"`, `"human"`, or `"rgb_array"`.
- `map_name` (String, default varies by registration): map preset name (e.g. `"4x4"`, `"8x8"`).
- `is_slippery` (Bool, default `true`): if `true`, the chosen action may slip to a side action.
- `desc` ([String], optional): custom map layout (rows of characters).

 
