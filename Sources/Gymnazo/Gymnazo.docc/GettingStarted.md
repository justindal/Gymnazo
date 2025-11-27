# Getting Started

Learn how to create and interact with Gymnazo environments.

## Overview

Gymnazo follows the OpenAI Gymnasium API, making it familiar to anyone who has used the Python library.

## Creating an Environment

Use the `Gymnazo` registry to create environments by their ID:

```swift
import Gymnazo

// Start the registry
Gymnazo.start()

// Create an environment
var env = Gymnazo.make("CartPole-v1")
```

## The Environment Loop

Interact with environments using the standard `reset` and `step` methods:

```swift
import MLX

// Reset the environment
var (observation, info) = env.reset()
var key = MLX.key(42)

var totalReward = 0.0
var done = false

while !done {
    // Sample a random action
    let action = env.action_space.sample(key: key)

    // Take a step
    let (nextObs, reward, terminated, truncated, stepInfo) = env.step(action)

    totalReward += reward
    observation = nextObs
    done = terminated || truncated
}

print("Episode finished with reward: \(totalReward)")
```

## Available Environments

Gymnazo includes several classic control and toy text environments:

| Environment           | ID                         | Description                 |
| --------------------- | -------------------------- | --------------------------- |
| CartPole              | `CartPole-v1`              | Balance a pole on a cart    |
| MountainCar           | `MountainCar-v0`           | Drive a car up a hill       |
| MountainCarContinuous | `MountainCarContinuous-v0` | Continuous action version   |
| Acrobot               | `Acrobot-v1`               | Swing up a two-link robot   |
| FrozenLake            | `FrozenLake-v1`            | Navigate a frozen lake grid |

## Using RL Agents

Gymnazo includes implementations of common RL algorithms:

```swift
import Gymnazo
import MLX

// Create a Q-Learning agent
let agent = QLearning(
    nStates: 16,
    nActions: 4,
    learningRate: 0.1,
    gamma: 0.99,
    epsilon: 1.0
)

// Train the agent
var env = Gymnazo.make("FrozenLake-v1") as! FrozenLake
var key = MLX.key(42)

for episode in 0..<1000 {
    var (state, _) = env.reset()
    var done = false

    while !done {
        let action = agent.chooseAction(
            actionSpace: env.action_space,
            state: state,
            key: &key
        )

        let (nextState, reward, terminated, truncated, _) = env.step(action)

        agent.update(
            state: state,
            action: action,
            reward: Float(reward),
            nextState: nextState,
            nextAction: 0,
            terminated: terminated
        )

        state = nextState
        done = terminated || truncated
    }

    // Decay epsilon
    agent.epsilon *= 0.995
}
```
