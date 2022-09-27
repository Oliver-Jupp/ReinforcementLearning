import torch as T
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim


class deepNeuralNet(nn.Module):

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, activation_function=relu):
        super(deepNeuralNet, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.activation_function = activation_function

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.activation_function(self.fc1(state))
        x = self.activation_function(self.fc2(x))
        actions = self.fc3(x)

        # For learning about activation functions
        # We don't "activate" it because we want the raw estimate
        # we don't want to use the relu functions, because this estimate
        # could be negative, could also be greater than 1, so sigmoid(??)
        # or a tan hyperbolic(??)

        return actions

    def getName(self):
        return self._get_name()
