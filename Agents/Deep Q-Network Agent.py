from Networks.DeepQNetwork import DeepQNetwork
import torch as T
import numpy as np


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self, lr, n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        # Memory storage, some people use deque
        # we're using named arrays

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

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
