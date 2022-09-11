import os
from os import rename, path, system

import gym
from gym.wrappers import RecordVideo
from Agents.Deep_QNetwork_Agent import Agent
import numpy as np


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
        print("Iteration:", i, "\t| Score: %2.f" % score, "\t| Average: %2.f" % totAvg, "\t| Avg (last 100): %2.f" % Avg100,
              "\t| Avg (last 50): %2.f" % Avg50)
        if Episode_trigger(i):
            oldPath = path.join(recordingFolder, "rl-video-episode-{}.mp4".format(str(i)))
            newPath = path.join(recordingFolder, "episode-{}-score={}.mp4".format(str(i), str(score)))
            rename(oldPath, newPath)

            old = path.join(recordingFolder, "rl-video-episode-{}.meta.json".format(str(i)))
            new = path.join(recordingFolder, "metaFiles")
            if not path.exists(new):
                os.mkdir(new)
            print("Moving meta file:")
            system("move {} {}".format(old, new))


    print("Saving model")
    agent.save()

    return scores


agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8], lr=0.001)
environment = gym.make("LunarLander-v2", new_step_api=True)
recordingFolder = "C:\\Users\\olive\\Downloads\\ReinforcementLearningVideos"
numberOfGames = 2000
percentRecording = 0.01

scores = recordFullLearning(agent, environment, recordingFolder, numberOfGames, percentRecording)




# WAS TEST.PY
"""
import os

import gym
import numpy as np
from os import system, path, mkdir
from Agents.Deep_QNetwork_Agent import Agent
from gym.wrappers import RecordVideo

def evaluatePerformance(agent):
    env = gym.make("LunarLander-v2", new_step_api=True, render_mode='human')

    agent.load()
    agent.evaluationMode()

    scores = []
    n_games = 100
    # Evaluate performance
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()

        stepCounter = 1
        while not done:
            stepCounter += 1

            action = agent.choose_action(observation)

            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            observation = observation_
            if stepCounter > 2499:
                print("Took 2500 steps")
                break
        scores.append(score)

        avg_score = np.mean(scores)

        print("Episode:", i, "\t| Score: %.2f" % score, "\t| Average Score: %.2f" % avg_score)

    return scores


def train(agent):
    env = gym.make("LunarLander-v2", new_step_api=True)
    agent.load()

    scores = []
    n_games = 500
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()

        step_counter = 1

        while not done:
            step_counter += 1
            if i > 480:
                env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            agent.store_transitions(observation, action, reward,
                                    observation_, done)
            agent.learn()
            observation = observation_

            if step_counter > 2499:
                print("Episode taking more than 2500 steps, aborting")
                break
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        full_avg_score = np.mean(scores)

        print("Episode: ", i, "\t| score %.2f" % score, "\t| average score (over last 100 games) %2.f" % avg_score,
              "\t| average score %2.f" % full_avg_score)

    print("Saving model")
    agent.save()


def record(agent):
    from gym.wrappers import RecordVideo
    env = gym.make("LunarLander-v2")
    env = RecordVideo(env, video_folder="savedModels\\Deep_QNetwork_Agent\\video")

    agent.load()
    agent.evaluationMode()

    scores = []
    n_games = 100
    # Evaluate performance
    for i in range(n_games):
        if i == 30:
            env.start_video_recorder()
        score = 0
        done = False
        observation = env.reset()

        stepCounter = 1
        while not done:
            stepCounter += 1

            action = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            score += reward
            observation = observation_
            if stepCounter > 2499:
                print("Took 2500 steps")
                break
        scores.append(score)

        avg_score = np.mean(scores)

        print("Episode:", i, "\t| Score: %.2f" % score, "\t| Average Score: %.2f" % avg_score)
        if i == 30:
            env.close_video_recorder()
            print("Recorded")

    return scores


record(Agent())
"""