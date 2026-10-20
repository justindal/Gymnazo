# Metro Run capture and native comparison

These tools target the supplied Subway Surfers iOS 3.70.1 build 96703,
UnityFramework UUID `4f9aeb54-055f-3708-89f0-6405a08627f9`.
`target_identity.json` retains the supplied build identity. Other builds require
new offsets and fingerprints. No original binary or game assets are included.

The capture helper and probe are preserved from the updated Metro_Run project.
They attach to an explicitly selected, already-running authorized game process.
They do not start the game, inject input, or write game state. Frida is an optional
external dependency, not a Gymnazo runtime dependency. Probe measurements and
original-device execution remain unverified. No original-game parity is claimed.

## Capture

On an authorized iPhone that already exposes the game through Frida:

```sh
python3 Tools/MetroRunner/capture_original.py --device DEVICE_ID --pid GAME_PID \
  --scenario flat_jump_110 --seconds 12 --output captures/original_jump.jsonl
```

The probe checks the target UUID and hook fingerprints before installing hooks.
Start grounded at track height zero and centered, with no powers, on a verified empty flat track.
Perform one short motion scenario, avoiding origin resets and run transitions.
Validate stationary/straight-running measurements before interpreting jumps.
The prepared probe does not capture all contacts, power state, layout, profile,
or board use. Hooks can affect timing. Never treat a capture file alone as proof.

## Build and replay the native Swift fixture

Build on macOS Apple Silicon using Xcode so MLX's Metal resources are present:

```sh
xcodebuild -workspace .swiftpm/xcode/package.xcworkspace -scheme GymnazoProfiler \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath /tmp/gymnazo-replay build
/tmp/gymnazo-replay/Build/Products/Debug/GymnazoProfiler metro-replay \
  captures/original_jump.jsonl --flat-track --output captures/swift_jump.jsonl
python3 Tools/MetroRunner/compare_traces.py captures/original_jump.jsonl \
  captures/swift_jump.jsonl --output captures/jump_comparison.json
```

Replay uses the actual native environment on an empty-track fixture with the
first captured position/time. Confirming `--flat-track` is required. Unsupported
build IDs, malformed motion, gaps, resets, active flight, conflicting impulses,
and overwriting a candidate output are rejected. Replay does not reconstruct
Unity routes from a seed or support powered/contact scenarios.

The comparator aligns time at the first sample, without a position shift. It
checks full coverage, finite motion, monotonic times, gaps under 100 ms, flags,
and attempted input timing. Default maximum errors are x/y 0.5, z 2, vy 5 and vz 3,
with 25 ms action tolerance. These are investigation thresholds, not calibrated
fidelity criteria. Set all five limits with `--tolerances limits.json` before an
experiment. Comparisons return a nonzero status on failure and cannot use a
simulator trace as an original-game reference. A passed motion scenario would
still not establish full-game fidelity.
