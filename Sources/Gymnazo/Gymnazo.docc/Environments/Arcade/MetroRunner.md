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

The five actions are no input, left, right, jump, and roll/dive
(indices 0 through 4). Hoverboards are excluded from the native environment.
One command is applied on the first physics tick of
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
    reward: MetroRunnerReward(distanceScale: 0.01, coinWeight: 0.05, deathPenalty: 5),
    powerLevel: 0,
    spawnProfile: MetroRunnerSpawnProfile()
)
var environment = try await Gymnazo.make(
    "MetroRunner-v0", options: ["config": config]
)
```

The curricula are `barriers`, `static`, `moving`, and `full`. They filter
entities at the original layout anchors, as the Python training adapter does.
`section` can select a fixed section, such as `Section_1_Track`, for repeated
diagnostic runs. Unknown sections and unsupported options throw.

`powerLevel` selects exported upgrade levels 0 through 6. Level zero uses
10-second magnet/sneakers/double-score powers and an 8-second jetpack; level six
uses 30 seconds and 25 seconds respectively.

The explicit offline spawn profile unlocks Magnet and ScoreDoubler and allows
mystery boxes. Pass `MetroRunnerSpawnProfile(unlockedFeatures: [], lootBoxFriendly: false)`
for the restricted set. Normal Easy anchors use integer weights 2/29/26/14/30/29
for key/magnet/sneakers/jetpack/mystery/double score. Group spacing is at least
2,400 units; key/jetpack/mystery repeats require 4,800/8,000/57,600 units. The
first spawn has no fabricated earliest-distance restriction. Medium/Special
anchors and unknown account requirements do not grant normal pickup eligibility.
Event, mission and server-controlled spawn sets remain excluded.

Reward is distance gained times `distanceScale`, plus coins gained times
`coinWeight`, minus `deathPenalty` on a crash. `info` includes the individual
reward components, distance, coins, score, death reason, events, physics steps,
curriculum, observation version, power level, spawn profile, keys, mystery-box count,
and `original_game_validated: false`. Mystery boxes are counted without fabricated
immediate rewards.

## Observations and training

The typed environment exposes the observation-version-five dictionary:

| Key | Shape | Meaning |
| --- | --- | --- |
| `player` | 41 | Motor state, powers, jump eligibility, dive, flight and custom-jump state |
| `entities` | 64 × 31 | Twenty-one type indicators and ten geometry/motion features |
| `entity_mask` | 64 | True for populated entity rows; false for padding |

Values are bounded to `[-1, 1]`. Entities within the 600-unit horizon are ranked
by time to contact, including overlapping bodies. Up to 48 rows are reserved for
hazards/support and 16 for pickups, filling unused capacity from the remainder.
Route identifiers and future random choices are absent.

The one-hot order is train, moving train, jump, roll, gate, ramp, coin, magnet,
sneakers, multiplier, jetpack, mystery, key, platform, pillar, signal, dumpster,
bush, powerbox, pogo, unknown. Geometry columns are x/40, width/80, front/2400,
rear/2400, collision bottom/200, collision top/200, relative velocity/1320,
support height/150, ramp rise length/2400, and airborne-pickup flag. Beam bounds
include the open space beneath the roll barrier. Moving train speeds remain
distinguishable during flight.

Player columns include actual forward
speed/440, y and height above support/200, vertical velocity/500, duration-relative
power timers, strict jump grace, separate diving state,
flight start height/ascent progress, custom-jump source/progress/duration/
height/captured speed, and captured flight speed. These are finite-horizon
observations, not a full-state Markov guarantee.

Earlier Metro Run checkpoints are incompatible: hoverboard actions and four
board-related player features were removed. Retrain agents. ExploreRL saves an observation-version tag and
rejects earlier Metro checkpoints with an explanation instead of loading their
weights into the new policy input. Other environments' session formats remain
readable.

The compatibility `Env` API exposes a Box and returns one array in sorted key
order: `entities`, `entity_mask`, `player` (2,089 floats). The binary mask is
encoded as float zeros and ones. For Gymnazo's typed vector trainers, use the provided
`flattened()` adapter:

```swift
let environment = try MetroRunnerValueEnvironment().flattened()
let model = try PPO(
    observationSpace: Box(low: -1, high: 1, shape: [MetroRunnerObservation.flatCount]),
    actionSpace: Discrete(n: MetroRunnerAction.allCases.count),
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
independent RNG streams; normal pickup arbitration has its own stream. Coin
counts do not consume route randomness. Motor and synthetic ramp traces are checked against
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
colliders, including pillars, signals, dumpsters, bushes and power boxes.
Unknown `obstacle_*` assets fail catalog validation instead of disappearing.

Coin lines/arcs consume the exported unweighted Hermite curves, native count
and speed/pivot scaling. Sneakers use their recovered custom ascent and recover
stumble. Pogo uses the explicit type-105 override, custom ascent to y160 over
300 forward units, 15 exclusive rows in three lanes, and a weighted airborne
reward. Pogo ascent disallows ordinary jumps; its dive starts
at -250 units/second. Jetpack captures twice the activation-time running speed,
ascends by smoothstep over 1.5 seconds to y95, and creates adjacent-lane coin paths
with halfway transition coins. Flyover filtering uses anchor preservation distance
and a strict landing boundary.

Exact controller contacts, corner/ceiling responses, train-gap insertion, sneaker
jump magnet, pogo cleanup/apex notifications, moving
coin attachments, original scoring/economy and Unity RNG remain unresolved.
Jetpack landing uses a nominal duration-times-speed endpoint and eagerly generated
coins. These are reconstruction limitations, not original-game-verified behavior.
No parity with or policy transfer to the original game is established.

The source data is a configuration export described by Metro_Run as coming
from Subway Surfers iOS 3.70.1 build 96703. No original executable, meshes,
textures, or audio are included. The catalog stores SHA-256 hashes of its four
source files, including gameplay configuration and geometry provenance. Regenerate it from the supplied `data` directory with:

```sh
python3 Tools/import_metro_catalog.py /path/to/Metro_Run/data Sources/Gymnazo/Resources/metro-runner.json
```

The full `train_geometry.json` audit export and unused game-mode schedules are
not runtime dependencies. The importer retains only the default scheduler,
core sections, curves, power/spawn configuration, and fields used by the simulator,
including random lane/mirror choices and exclusive random-child groups.

## Validation

Run the native integration tests on Apple Silicon with Xcode, including MLX's
Metal shaders:

```sh
xcodebuild -workspace .swiftpm/xcode/package.xcworkspace \
  -scheme Gymnazo-Package -destination 'platform=macOS,arch=arm64' \
  -parallel-testing-enabled NO -only-testing:GymnazoTests/MetroRunnerTests \
  -only-testing:GymnazoTests/MetroRunnerFidelityTests \
  -only-testing:GymnazoTests/MetroRunnerTraceTests test
```

The focused suite checks catalog coverage, scheduler boundaries and connections,
geometry, Python motor/ramp reference traces, single-command frame skipping,
time limits, crashes without protective saves, pickups, observation support/masks,
curricula, deterministic bounded long rollouts, both environment APIs,
parallel collection, and a PPO parameter update.

## Original-game capture and motion comparison

`Tools/MetroRunner` contains the supplied build-guarded Frida capture probe,
comparison tools, and target identity. The original executable is not included.
The probe is syntax-checked and trace infrastructure is tested; original target
execution and paired original-game comparisons remain unverified.

Use the Xcode-built `GymnazoProfiler metro-replay ORIGINAL.jsonl --flat-track
--output SIMULATOR.jsonl` command for native Swift clear-track replay. It requires
a centered, grounded, unpowered start at track height zero, known target UUID, continuous finite
samples and at most one attempted impulse per 60 Hz tick. It retains the first
sample's position/time and refuses to overwrite an existing output. The strict
comparator checks full coverage, positions/velocities, grounded/flight flags and
attempted action timing. Synthetic tests of the codec are not original-game
validation. See `Tools/MetroRunner/README.md` for commands and scope.
