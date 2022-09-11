import gym
import numpy as np
from Agents.Deep_QNetwork_Agent import Agent

if __name__ == '__main__':
    env = gym.make("LunarLander-v2", new_step_api=True)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8], lr=0.003)

    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            if i > 400:
                env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            agent.store_transitions(observation, action, reward,
                                    observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print("Episode,", i, "score %.2f" % score,
              "average score %2.f" % avg_score,
              "epsilon %.2f" % agent.epsilon)

    print("Saving model")
    agent.save()
else:
    print("Not ran as __main__")