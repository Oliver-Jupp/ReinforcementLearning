from Networks.Deep_QNetwork import Deep_QNetwork
import torch as T
import numpy as np
import os
import pickle


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01,
                 eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = Deep_QNetwork(lr, input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions)

        # Memory storage, some people use deque
        # we're using named arrays

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transitions(self, state, action, reward, state_, done):

        # What is the position of the first unoccupied memory
        # Using the modulus here gives the property that this will wrap around
        # so after 999999 items, then it will start re-writing the stuff from the beginning
        index = self.mem_cntr % self.mem_size

        # So now we know where to store it, we store it
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        # Memory index has been filled, so we increase
        self.mem_cntr += 1

    # The agent has to have a function so that it can choose an action
    # and that will be based on the observation of the current state of the environment
    def choose_action(self, observation):

        # So here we're dealing with the exploration|exploitation problem,
        # and we're dealing with it normally. E.g.
        # we've got epsilon, a value that starts out at 1, meaning it will always take a random action
        # this then goes down more, so we can take more greedy actions etc. etc., until a given
        # minimum epsilon.
        if np.random.random() > self.epsilon:

            # Take our observation, turn it into a tensor
            state = T.tensor([observation]).to(self.Q_eval.device)
            # Pass the state through our network
            actions = self.Q_eval.forward(state)
            # Get the argmax to return the integer that corresponds to the maximal action for given state
            action = T.argmax(actions).item()

        # Then take random (not greedy) action
        else:
            action = np.random.choice(self.action_space)

        return action

    # We need the agent to learn from its experiences

    # If memory is filled with zeros, we can either let agent play a bunch of games randomly, until
    # it fills up its memory, and then you can start learning
    # Or
    # Simply start learning as soon as you have filled up a batch_size of memory
    # ^ this is what we're doing
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        # Zeroing the gradient on our optimizer
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # What we're doing here is we are converting the numpy array subset of ur agents memory into a pytorch tensor
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        # we want to get the values of the actions of we took
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)

        q_next[terminal_batch] = 0.0

        # Maximum value of the next state (purely greedy action)
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        # Loss then is:
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        # Decrement epsilon
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min

    def save(self):
        folderName = os.path.join("savedModels", str(__name__).split(".")[1])

        if not os.path.exists(folderName):
            os.mkdir("savedModels")
            os.mkdir(folderName)

        T.save(self.Q_eval, os.path.join(folderName, "model.pt"))

        # This may be a very hacky-way to do this, but i find it the easiest... so
        # The idea is that we use the f"{self.variable=}" to get the return name: "self.variable=value"
        # then we use .split to split the string between the . and the = to get the variable name
        listOfItemsToSave = [[f"{self.gamma=}".split(".")[1].split("=")[0], self.gamma],
                             [f"{self.epsilon=}".split(".")[1].split("=")[0], self.epsilon],
                             [f"{self.eps_min=}".split(".")[1].split("=")[0], self.eps_min],
                             [f"{self.eps_dec=}".split(".")[1].split("=")[0], self.eps_dec],
                             [f"{self.lr=}".split(".")[1].split("=")[0], self.lr],
                             [f"{self.action_space=}".split(".")[1].split("=")[0], self.action_space],
                             [f"{self.mem_size=}".split(".")[1].split("=")[0], self.mem_size],
                             [f"{self.batch_size=}".split(".")[1].split("=")[0], self.batch_size],
                             [f"{self.mem_cntr=}".split(".")[1].split("=")[0], self.mem_cntr],
                             [f"{self.state_memory=}".split(".")[1].split("=")[0], self.state_memory],
                             [f"{self.new_state_memory=}".split(".")[1].split("=")[0], self.new_state_memory],
                             [f"{self.action_memory=}".split(".")[1].split("=")[0], self.action_memory],
                             [f"{self.reward_memory=}".split(".")[1].split("=")[0], self.reward_memory],
                             [f"{self.terminal_memory=}".split(".")[1].split("=")[0], self.terminal_memory]]

        for item in listOfItemsToSave:
            with open(os.path.join(folderName, item[0] + ".pickle"), mode="wb") as file:
                pickle.dump(item[1], file)

        print("\n\n")
        return True

    def load(self):
        folderName = os.path.join("savedModels", str(__name__).split(".")[1])

        if not os.path.exists(folderName):
            print("Found no existing model to load")
            return False

        self.Q_eval = T.load(os.path.join(folderName, "model.pt"))

        listOfItemsToLoad = [f"{self.gamma=}".split(".")[1].split("=")[0],
                             f"{self.epsilon=}".split(".")[1].split("=")[0],
                             f"{self.eps_min=}".split(".")[1].split("=")[0],
                             f"{self.eps_dec=}".split(".")[1].split("=")[0],
                             f"{self.lr=}".split(".")[1].split("=")[0],
                             f"{self.action_space=}".split(".")[1].split("=")[0],
                             f"{self.mem_size=}".split(".")[1].split("=")[0],
                             f"{self.batch_size=}".split(".")[1].split("=")[0],
                             f"{self.mem_cntr=}".split(".")[1].split("=")[0],
                             f"{self.state_memory=}".split(".")[1].split("=")[0],
                             f"{self.new_state_memory=}".split(".")[1].split("=")[0],
                             f"{self.action_memory=}".split(".")[1].split("=")[0],
                             f"{self.reward_memory=}".split(".")[1].split("=")[0],
                             f"{self.terminal_memory=}".split(".")[1].split("=")[0]]

        for item in listOfItemsToLoad:
            fileName = os.path.join(folderName, item + ".pickle")
            if not os.path.exists(fileName):
                print("File:", fileName, "does not exist.")
                return False

            with open(fileName, mode="rb") as file:
                value = pickle.load(file)

                if item == "gamma":
                    self.gamma = value
                elif item == "epsilon":
                    self.epsilon = value
                elif item == "eps_min":
                    self.eps_min = value
                elif item == "eps_dec":
                    self.eps_dec = value
                elif item == "lr"
                    self.lr = value
                elif item == "action_space":
                    self.action_space = value
                elif item == "mem_size":
                    self.mem_size = value
                elif item == "batch_size":
                    self.batch_size = value
                elif item == "mem_cntr":
                    self.mem_cntr = value
                elif item == "state_memory":
                    self.state_memory = value
                elif item == "new_state_memory":
                    self.new_state_memory = value
                elif item == "action_memory":
                    self.action_memory = value
                elif item == "reward_memory":
                    self.reward_memory = value
                else:
                    self.terminal_memory = value

            return True


# https://deeplizard.com/learn/video/HGeI30uATws/
# https://www.youtube.com/watch?time_continue=4&v=HGeI30uATws&feature=emb_title/
# or maybe this: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
