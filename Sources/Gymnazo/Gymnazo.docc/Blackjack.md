# Blackjack

A simple card game environment (ToyText).

## Overview

Try to beat the dealer by getting a hand total closer to 21 without going over.

## Action Space

Two discrete actions:

| Action | Meaning |
|--------|---------|
| 0      | Stick   |
| 1      | Hit     |

## Observation Space

The observation is a ``BlackjackObservation``:

- `playerSum`: the player’s current hand sum
- `dealerCard`: the dealer’s showing card (1–10, where 1 is Ace)
- `usableAce`: 0/1 indicating if the player has a usable Ace

## Rewards

- Win: `+1`
- Lose: `-1`
- Draw: `0`

If `natural` is enabled, winning with a natural blackjack may yield `+1.5` (see rules in the implementation).

## Episode Termination

Terminates when the player busts or when the player sticks and the dealer finishes their draw.

## Arguments

```swift
let env = make("Blackjack", kwargs: [
    "render_mode": "human",
    "natural": false,
    "sab": false
])
```

- `render_mode` (String, optional): `"human"` or `"rgb_array"`.
- `natural` (Bool, default `false`): whether to pay out additional reward for a natural blackjack.
- `sab` (Bool, default `false`): Sutton & Barto rules; if `true`, `natural` is ignored.

## Topics

### Environment Types

- ``Blackjack``

### Supporting Types

- ``BlackjackObservation``


