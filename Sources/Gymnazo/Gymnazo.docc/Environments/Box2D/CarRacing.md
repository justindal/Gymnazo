# Car Racing

A top-down racing environment from the Box2D suite.

## Overview

The easiest control task to learn from pixels - a top-down racing environment. The generated track is random every episode.

Some indicators are shown at the bottom of the window along with the state RGB buffer. From left to right: true speed, four ABS sensors, steering wheel position, and gyroscope.

Remember: it's a powerful rear-wheel drive car - don't press the accelerator and turn at the same time.

## Action Space

### Continuous (CarRacing)

There are 3 continuous actions:

| Index | Action   | Range                             |
| ----- | -------- | --------------------------------- |
| 0     | Steering | -1 (full left) to +1 (full right) |
| 1     | Gas      | 0 to 1                            |
| 2     | Braking  | 0 to 1                            |

### Discrete (CarRacingDiscrete)

There are 5 discrete actions:

| Action | Description |
| ------ | ----------- |
| 0      | Do nothing  |
| 1      | Steer right |
| 2      | Steer left  |
| 3      | Gas         |
| 4      | Brake       |

## Observation Space

A top-down 96x96 RGB image of the car and race track.

- **Shape**: `[96, 96, 3]`
- **Type**: `uint8`
- **Range**: 0 to 255

## Rewards

The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1\*732 = 926.8 points.

## Starting State

The car starts at rest in the center of the road.

## Episode Termination

The episode finishes when all the tiles are visited. The car can also go outside the playfield - that is, far off the track, in which case it will receive -100 reward and die.

## Arguments

```swift
import Gymnazo

let env = await Gymnazo.make("CarRacing", options: [
    "render_mode": "rgb_array",
    "lap_complete_percent": 0.95,
    "domain_randomize": false
])
```

- `lap_complete_percent` (Float, default 0.95): dictates the percentage of tiles that must be visited by the agent before a lap is considered complete.
- `domain_randomize` (Bool, default false): enables the domain randomized variant of the environment. In this scenario, the background and track colours are different on every reset.

For discrete actions, use `CarRacingDiscrete` instead:

```swift
import Gymnazo

let env = await Gymnazo.make("CarRacingDiscrete", options: [
    "render_mode": "rgb_array"
])
```

## Reset Arguments

Passing the option `options["randomize"] = true` will change the current colour of the environment on demand. Correspondingly, passing the option `options["randomize"] = false` will not change the current colour of the environment. `domain_randomize` must be `true` on init for this argument to work.

```swift
var env = CarRacing(domainRandomize: true)

// Normal reset - changes colour scheme by default
let obs1 = env.reset()

// Reset with colour scheme change
let obs2 = env.reset(options: ["randomize": true])

// Reset with no colour scheme change
let obs3 = env.reset(options: ["randomize": false])
```

## Version History

- v1: Swift port using Box2D physics engine (Gymnazo)
- v2: Change truncation to termination when finishing the lap (Gymnasium 1.0.0)
- v1: Change track completion logic and add domain randomization (Gymnasium 0.24.0)
- v0: Original version

## References

- Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car

## Credits

Original environment created by Oleg Klimov. Swift port by Gymnazo.

## Topics

### Supporting Types

- ``CarRacingSnapshot``
- ``Car``
- ``RoadTile``
- ``TrackData``
