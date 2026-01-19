# CliffWalking

A grid-world task where stepping into the “cliff” yields a large penalty (ToyText).

## Overview

The agent starts at `S` and must reach `G`. The bottom row between them is a cliff. Stepping into the cliff gives a large negative reward and resets the agent back to `S`.

## Action Space

Four discrete actions:

| Action | Meaning |
|--------|---------|
| 0      | Up      |
| 1      | Right   |
| 2      | Down    |
| 3      | Left    |

## Observation Space

The observation is an `Int` representing the agent’s current state index in the grid.

## Rewards

As implemented in Gymnazo:

- `-1` for safe moves
- `-100` for stepping into the cliff (and the state is reset back to start)

## Episode Termination

Terminates only when the agent reaches the goal.

Stepping into the cliff **does not** terminate; it resets the state to the start position.

Truncation is typically handled by the default `maxEpisodeSteps` wrapper for the registered env.

## Arguments

```swift
import Gymnazo

let env = try await Gymnazo.make("CliffWalking", options: [
    "render_mode": "ansi"
])
```

- `render_mode` (String, optional): `"ansi"`, `"human"`, or `"rgb_array"`.

 
