import gym
env = gym.make("MountainCar-v0",new_step_api=True)

print("Action:\t\t\t",env.action_space)
print("Observation:\t",env.observation_space)

t = 0
while True:
    t += 1
    observation = env.reset()
    env.render()


"""
t = 0
while True:
    t += 1
    observation = env.reset()
    env.render()

    print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()
"""