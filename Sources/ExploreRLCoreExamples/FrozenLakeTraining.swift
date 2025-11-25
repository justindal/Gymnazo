import ExploreRLCore
import MLX

@main
struct FrozenLakeTrainingExample {
    static func main() {
        Gymnasium.start()

        var env = Gymnasium.make(
            "FrozenLake-v1",
            kwargs: ["is_slippery": true]
        ) as! FrozenLake

        let nStates = env.observation_space.n
        let nActions = env.action_space.n

        let agent = QLearningAgent(
            learningRate: 0.1,
            gamma: 0.99,
            stateSize: nStates,
            actionSize: nActions,
            epsilon: 0.1
        )

        var key = MLX.key(0)

        for episode in 0..<100 {
            var (obs, _) = env.reset(seed: UInt64(episode))
            var done = false
            var totalReward: Double = 0

            while !done {
                let action = agent.chooseAction(
                    actionSpace: env.action_space,
                    state: obs,
                    key: &key
                )

                let step = env.step(action)
                let nextObs = step.obs
                let reward = Float(step.reward)
                let terminated = step.terminated

                // Choose next action for the update (Q-learning ignores this, but API requires it)
                let nextAction = agent.chooseAction(
                    actionSpace: env.action_space,
                    state: nextObs,
                    key: &key
                )

                _ = agent.update(
                    state: obs,
                    action: action,
                    reward: reward,
                    nextState: nextObs,
                    nextAction: nextAction,
                    terminated: terminated
                )

                obs = nextObs
                totalReward += step.reward
                done = terminated || step.truncated
            }

            print("episode: \(episode), return: \(totalReward)")
        }
    }
}
