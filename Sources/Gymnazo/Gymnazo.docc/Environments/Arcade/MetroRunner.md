# Metro Run

`MetroRunner-v0` is a native Swift port of the supplied Metro_Run training
simulator. It runs on iOS 18+ and macOS 15+ without Python, Pygame, Gymnasium,
or Stable-Baselines3. Physics and route state are Swift values; MLX is used
at the policy boundary through Gymnazo's existing trainers.

```swift
import Gymnazo
import MLX

var environment = try await Gymnazo.make("MetroRunner-v0")
let reset = try environment.reset(seed: 42)
let step = try environment.step(MLXArray(Int32(3)))
```

The six actions are no input, left, right, jump, roll/dive, and hoverboard
(indices 0 through 5). One command is applied on the first physics tick of
each decision. The default advances four ticks at 60 Hz, stopping immediately
on a crash or the episode's tick limit. Crashes terminate; the default
300-second limit truncates. An environment must be reset before use and after
either end condition, following the [Gymnasium step contract](https://gymnasium.farama.org/api/env/).
Revival and invulnerability controls are unavailable in the training API.

## Configuration

```swift
let config = MetroRunnerConfig(
    curriculum: .barriers,
    frameSkip: 4,
    maxEpisodeSeconds: 60,
    reward: MetroRunnerReward(distanceScale: 0.01, coinWeight: 0.05, deathPenalty: 5)
)
var environment = try await Gymnazo.make(
    "MetroRunner-v0", options: ["config": config]
)
```

The curricula are `barriers`, `static`, `moving`, and `full`. They filter
entities at the original layout anchors, as the Python training adapter does.
`section` can select a fixed section, such as `Section_1_Track`, for repeated
diagnostic runs. Unknown sections and unsupported options throw.

Reward is distance gained times `distanceScale`, plus coins gained times
`coinWeight`, minus `deathPenalty` on a crash. `info` includes the individual
reward components, distance, coins, score, death reason, events, physics steps,
curriculum, and observation version.

## Observations and training

The typed environment preserves the observation-version-two dictionary:

| Key | Shape | Meaning |
| --- | --- | --- |
| `player` | 32 | Normalized motor state, support height, speed progress, powers and board availability |
| `entities` | 64 × 22 | Fourteen type indicators followed by normalized position, bounds, relative velocity and support height |
| `entity_mask` | 64 | True for populated entity rows; false for padding |

Values are bounded to `[-1, 1]`. Entities within the 600-unit horizon are
ranked by time to contact, including overlapping bodies. Up to 48 rows are
reserved for hazards/support and 16 for pickups, with unused rows filled from
the remainder. Route identifiers and future random choices are absent.

Entity types are ordered as `MetroRunnerEntityKind.allCases`: train, moving
train, jump barrier, roll barrier, gate, ramp, coin, magnet, sneakers,
multiplier, jetpack, mystery box, key, and platform. Geometry columns are
x/40, width/80, front distance/2400, rear distance/2400, bottom/150,
top/150, relative velocity/440, and support height/150.

Player columns follow Metro_Run's `PLAYER_FEATURES`: x/20, y/150, height above
support/150, vertical velocity/200, speed/220, target lane minus one, destination
lane minus one, origin lane minus one, lane-change fraction, direction,
grounded, support/150, roll remaining/0.6, time since landing/0.15, lane travel/20,
lane duration/0.27, speed-ramp progress, magnet/sneakers/multiplier/jetpack/board
remaining fractions, boards/10, invulnerability/3, stumble/5, intro chase fraction,
body height/9, jump height/40, gravity/200, minimum lane travel/1.5,
lane elapsed/0.27, and board availability.

The compatibility `Env` API exposes a Box and returns one array in sorted key
order: `entities`, `entity_mask`, `player` (1,504 floats). The binary mask is
encoded as float zeros and ones. For Gymnazo's typed vector trainers, use the provided
`flattened()` adapter:

```swift
let environment = try MetroRunnerValueEnvironment().flattened()
let model = try PPO(
    observationSpace: Box(low: -1, high: 1, shape: [MetroRunnerObservation.flatCount]),
    actionSpace: Discrete(n: 6),
    policyConfig: PPOPolicyConfig(netArch: .shared([128, 128]), featuresExtractor: .flatten),
    config: PPOConfig(nSteps: 256, batchSize: 64, nEpochs: 4),
    seed: 42
)
try await model.learn(environments: [environment], totalTimesteps: 4096)
```

This is a training example, not evidence of a capable policy. The tests use a
short PPO run only to verify that native collection updates policy parameters.
Registry-created typed environments also support actor-owned parallel vectors.

## Seeds, rendering, and fidelity

`reset(seed:)` reproduces a Swift episode, including subsequent unseeded-reset
sequences. `reset()` advances a local seed generator. `route_seed` in `info` is
a decimal string preserving all 64 bits; pass it back as a reset option to
replay a route. Reset options also accept a curriculum string.

Swift uses Gymnazo's `SeededGenerator`, so Python/NumPy/Unity seed sequences and
saved SB3 policies are not interchangeable. Routes and gameplay effects use
independent RNG streams. Motor and synthetic ramp traces are checked against
values produced by the supplied Python simulator.

`currentSnapshot` is immutable and Sendable, including support height, ramp
geometry, and active powers. Human rendering returns that snapshot, and ANSI
rendering returns a short telemetry line. ExploreRL consumes snapshots in its
native perspective viewer for training and evaluation. Pygame sound, desktop
play controls, and Gymnazo RGB rendering are not ported. Metadata's suggested
15 FPS is for the default frame skip.

The bundled data preserves all 49 chunk hierarchies, 49 core sections, and 39
default weighted scheduler entries. Normal scheduling reaches 42 chunks; the
remaining context templates are accessible with `section`. Composite and
alternating order, strict distance bounds, priorities, uniqueness, and repeat
spacing are preserved. Runtime data includes recovered train/platform
colliders. Approximate ramp contacts, player collision bounds, random child
selection, spawn distributions, powers, scoring, and flight remain the supplied
simulator's approximations. No parity with or policy transfer to the original
game is established.

The source data is a configuration export described by Metro_Run as coming
from Subway Surfers iOS 3.70.1 build 96703. No original executable, meshes,
textures, or audio are included. The catalog stores SHA-256 hashes of its two
source files. Regenerate it from the supplied `data` directory with:

```sh
python3 Tools/import_metro_catalog.py /path/to/Metro_Run/data Sources/Gymnazo/Resources/metro-runner.json
```

The full `train_geometry.json` audit export and unused game-mode schedules are
not runtime dependencies. The importer retains only the default scheduler,
core sections, and fields used by the simulator, including random lane/mirror
choices and exclusive random-child groups.

## Validation

Run the native integration tests on Apple Silicon with Xcode, including MLX's
Metal shaders:

```sh
xcodebuild -workspace .swiftpm/xcode/package.xcworkspace \
  -scheme Gymnazo-Package -destination 'platform=macOS,arch=arm64' \
  -parallel-testing-enabled NO -only-testing:GymnazoTests/MetroRunnerTests test
```

The focused suite checks catalog coverage, scheduler boundaries and connections,
geometry, Python motor/ramp reference traces, single-command frame skipping,
time limits, crashes, board saves, pickups, observation support/masks,
curricula, deterministic bounded long rollouts, both environment APIs,
parallel collection, and a PPO parameter update.
