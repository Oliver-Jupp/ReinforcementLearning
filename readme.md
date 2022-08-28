The goal of reinforcement learning is to train an **_agent_** to complete a task within an uncertain **_environment_**.

At each time interval, the **_agent_** recieves **_observations_** and a **_reward_** from the **_environment_** and sends an **_action_** to the **_environment_**.

The **_agent_** contains two components:

- Policy
  - A mapping that selects **_actions_** based on the **_observations_** from the **_environment_**. Typically, the **_policy_** is a function approximator with tunable parameters, such as a deep neural network.
- Learning Algorithm
  - The algorithm continuously updates the policy parameters based on the **_actions_**, **_observations_**, and **_rewards_**. The goal of the algorithm is to find an optimal policy that maximises the expected cumulative long-term **_reward_** recieved during the task.

  
With the agents I know, (Q_Learning + Deep Q-Network) they are both Value-based **_agents_** with a discreet **_action space_**, however there are more types of **_agents_**, e.g:

| Agent                                                        | Type         | Action Space           |
|--------------------------------------------------------------|--------------|------------------------|
| Q-Learning Agents (Q)                                        | Value-Based  | Discrete               |
| Deep Q-Network Agents (DQN)                                  | Value-Based  | Discrete               |
| SARSA Agents                                                 | Value-Based  | Discrete               |
| Policy Gradient Agents (PG)                                  | Policy-Based | Discrete or continuous |
| Actor-Critic Agents (AC)                                     | Actor-Critic | Discrete or continuous |
| Proximal Policy Optimization Agents (PPO)                    | Actor-Critic | Discrete or continuous |
| Trust Region Policy Optimization Agents (DDPG)               | Actor-Critic | Discrete or continuous |
| Deep Deterministic Policy Gradient Agents (DDPG)             | Actor-Critic | Continuous             |
| Twin-Delayed Deep Deterministic Policy Gradient Agents (TD3) | Actor-Critic | Continuous             |
| Soft Actor-Critic Agents (SAC)                               | Actor-Critic | Continuous             |


[I found these here](https://www.mathworks.com/help/reinforcement-learning/ug/create-agents-for-reinforcement-learning.html)