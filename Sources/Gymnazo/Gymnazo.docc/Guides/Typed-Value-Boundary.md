# Typed Value Boundary

Keep environment state and concurrency transfers in Swift values, and convert to `MLXArray` only when a policy or training operation needs tensors.

## Overview

The typed API uses ``ValueEnvironment`` for scalar simulation, ``ValueSpace`` for typed contracts, and ``SpaceValue`` for type-erased values. These values conform to `Sendable`, so they can move between actors without transferring MLX execution state.

The compatibility ``Env`` API remains available for existing integrations. New collection and training code should prefer the typed boundary when it does not need compatibility rendering.

## Create a typed environment

Use ``Gymnazo/makeValue(_:maxEpisodeSteps:options:)`` for a registered environment:

```swift
import Gymnazo

let environment = try await Gymnazo.makeValue("CartPole")
let reset = try await environment.reset(seed: 42)
let step = try await environment.step(.discrete(0))

print(reset.observation)
print(step.reward, step.terminated, step.truncated)
try await environment.close()
```

``AnyValueEnvironment`` owns the underlying environment actor and validates actions and observations against its ``AnyValueSpace`` contracts. Episode limits preserve terminal observations and add `TimeLimit.truncated` to the typed info dictionary.

## Collect from typed vectors

Use ``Gymnazo/makeValueVec(_:numEnvs:mode:autoresetMode:maxEpisodeSteps:options:)`` for registered vector environments:

```swift
let vector = try await Gymnazo.makeValueVec(
    "CartPole",
    numEnvs: 4,
    mode: .parallel,
    autoresetMode: .sameStep
)

_ = try await vector.reset(seed: 42)
let result = try await vector.step([
    .discrete(0), .discrete(1), .discrete(0), .discrete(1)
])

for index in result.rewards.indices {
    print(result.rewards[index], result.terminations[index])
    if let final = result.finalObservations[index] {
        print("terminal observation:", final)
    }
}

try await vector.close()
```

`.sync` runs typed environments sequentially. `.parallel` gives each environment its own actor and uses structured task groups for reset and step. ``ValueVectorStep`` keeps current observations, final observations, final info, and reset info in separate fields.

## Collect with one batched policy boundary

For a homogeneous ``ValueEnvironment``, ``ValueCollectionSession`` owns the vector and prevents overlapping collection sessions:

```swift
let session = try ValueCollectionSession(
    environments: [
        try CartPoleValueEnvironment(),
        try CartPoleValueEnvironment(),
    ]
)

let collection = try await session.collect(steps: 128, seed: 42) { observations in
    observations.map { _ in 0 }
}

print(collection.frames.count)
try await session.close()
```

Algorithm helpers such as ``DQN/collect(from:steps:seed:deterministic:)`` and ``PPO/collect(from:steps:seed:deterministic:)`` invoke a batched `act` boundary. Tensorization stays inside the algorithm path rather than in each environment step.

## Compose typed wrappers

Typed wrappers preserve or explicitly derive their output spaces:

```swift
var environment = try CartPoleValueEnvironment()
    .timeLimited(200)
    .observationsDelayed(by: 2)
    .rewardsClipped(min: -1, max: 1)
```

Use the chainable wrapper helpers `observationsReshaped(to:)`, `maxAndSkip(_:)`, `stickyActions(probability:seed:)`, and `filteringObservation(keys:)`. They construct ``ReshapeObservationValueEnvironment``, ``MaxAndSkipObservationValueEnvironment``, ``StickyActionValueEnvironment``, and ``FilterDictionaryObservationValueEnvironment`` values.

## Tensorize at the policy boundary

``TensorValue`` and ``MLXValueCodec`` provide explicit conversions for policy and training boundaries. Avoid converting each scalar physics value to an MLX tensor during environment simulation; batch typed observations once before inference or learning.

## Topics

- ``ValueEnvironment``
- ``ValueSpace``
- ``SpaceValue``
- ``AnyValueEnvironment``
- ``AnyValueVectorEnvironment``
- ``ValueCollectionSession``
- ``ReshapeObservationValueEnvironment``
- ``MaxAndSkipObservationValueEnvironment``
- ``StickyActionValueEnvironment``
- ``FilterDictionaryObservationValueEnvironment``
- ``TensorValue``
- ``MLXValueCodec``
