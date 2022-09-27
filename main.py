import gym
import numpy as np
from os import rename, path, system, mkdir
from gym.wrappers import RecordVideo
from torch.nn.functional import relu, leaky_relu, sigmoid, tanh, softmax
from Agents.Deep_QNetwork_Agent import Agent
from Networks.deepNeuralNet import deepNeuralNet


def recordFullLearning(agent, environment, recordingFolder, numberOfGames, percentRecording):
    listOfRecordingIndexes = [x for x in range(0, numberOfGames + 1, int(numberOfGames * percentRecording))]
    Episode_trigger = lambda x: x in listOfRecordingIndexes

    env = RecordVideo(environment, video_folder=recordingFolder, episode_trigger=Episode_trigger)
    print(len(listOfRecordingIndexes), "episodes queued for recording\nEpisodes queued:", listOfRecordingIndexes)

    scores = []

    env.reset()

    for i in range(numberOfGames):

        score = 0
        done = False
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transitions(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

        scores.append(score)

        totAvg = np.mean(scores)
        Avg100 = np.mean(scores[-100:])
        Avg50 = np.mean(scores[-50:])
        print("Iteration:", i, "\t| Score: %2.f" % score, "\t| Average: %2.f" % totAvg,
              "\t| Avg (last 100): %2.f" % Avg100,
              "\t| Avg (last 50): %2.f" % Avg50)
        if Episode_trigger(i):
            oldPath = path.join(recordingFolder, "rl-video-episode-{}.mp4".format(str(i)))
            newPath = path.join(recordingFolder, "episode-{}-score={}.mp4".format(str(i), str(score)))
            rename(oldPath, newPath)

            old = path.join(recordingFolder, "rl-video-episode-{}.meta.json".format(str(i)))
            new = path.join(recordingFolder, "metaFiles")
            if not path.exists(new):
                mkdir(new)
            print("Moving meta file:")
            system("move {} {}".format(old, new))

    print("Saving model")
    agent.save(scores)

    return scores


def learn(agent, environment, numberOfGames):
    scores = []

    environment.reset()

    for i in range(numberOfGames):

        score = 0
        done = False
        observation = environment.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = environment.step(action)
            score += reward
            agent.store_transitions(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

        scores.append(score)

        totAvg = np.mean(scores)
        Avg100 = np.mean(scores[-100:])
        Avg50 = np.mean(scores[-50:])
        print("Iteration:", i, "\t| Score: %2.f" % score, "\t| Average: %2.f" % totAvg,
              "\t| Avg (last 100): %2.f" % Avg100,
              "\t| Avg (last 50): %2.f" % Avg50)

    print("Saving model")
    agent.save(scores)

    return scores


# https://www.geeksforgeeks.org/activation-functions-in-pytorch/

activationFunctions = [relu, leaky_relu, sigmoid, tanh, softmax]
activationFunctionsScores = []
environment = gym.make("LunarLander-v2", new_step_api=True)
recordingFolder = "C:\\Users\\olive\\Downloads\\ReinforcementLearningVideos"
numberOfGames = 2000
percentRecording = 0.01

for func in activationFunctions:
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8], lr=0.001,
                  Q_eval=deepNeuralNet, activation_function=func)
    scores = recordFullLearning(agent, environment, recordingFolder, numberOfGames, percentRecording)
    activationFunctionsScores.append(scores)
