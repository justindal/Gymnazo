# Taxi

A classic grid-world pickup-and-dropoff task (ToyText).

## Overview

Pick up a passenger at one location and drop them off at the destination. The state encodes taxi position, passenger location, and destination.

## Action Space

Six discrete actions:

| Action | Meaning  |
|--------|----------|
| 0      | South    |
| 1      | North    |
| 2      | East     |
| 3      | West     |
| 4      | Pickup   |
| 5      | Dropoff  |

## Observation Space

The observation is an `Int` encoding the full state (Taxi position, passenger, destination).

## Rewards

Typical reward structure:

- `-1` per step
- `+20` for successful dropoff
- `-10` for illegal pickup/dropoff

## Episode Termination

Terminates when the passenger is successfully dropped off.

Truncation is typically handled by the default `maxEpisodeSteps` wrapper for the registered env.

## Arguments

```swift
let env = make("Taxi", kwargs: [
    "render_mode": "ansi"
])
```

- `render_mode` (String, optional): `"ansi"`, `"human"`, or `"rgb_array"`.

## Topics

### Environment Types

- ``Taxi``


