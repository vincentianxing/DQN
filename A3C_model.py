# A3C neural network architecture
# Vincent Zhu

import torch
from torch import nn
import torch.nn.functional as F


class A3C(nn.Module):
    def __init__(self, input_size=4, n_actions=4):
        super().__init__()

        self.conv = nn.Sequential(
            # Input: 84 * 84 * 4 produced by preprocessing
            nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Input: 20 * 20
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Input: 9 * 9
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Final hidden layer: fully-connected and consists of 512 rectifier units.
        # Input: 7 * 7
        self.final_layer = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.final_activation = nn.ReLU()

        # Output layer: a fully-connected linear layer with a single output for each valid action.
        # self.head = nn.Sequential(
        #     nn.Linear(in_features=512, out_features=256),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=n_actions),
        # )

        # Policy (actor)
        self.policy = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

        # Value (critic)
        self.value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.distribution = torch.distributions.Categorical

    def forward(self, x: torch.Tensor):
        # Use GPU
        # x = x.float().to('cuda') / 255

        # Convolutional layers
        x = self.conv(x)

        # Reshape for linear layers
        # x = x.view(x.size(0), -1)
        x = x.reshape((-1, 7 * 7 * 64))

        # Linear layer
        x = self.final_activation(self.final_layer(x))

        # Output layer
        logits = self.policy(x)
        values = self.value(x)

        return logits, values

    # Choose action based on model
    def choose_action(self, state):
        self.eval()
        logits, values = self.forward(state)
        # print(logits)
        prob = F.softmax(logits, dim=1).data
        # print(prob)
        m = self.distribution(prob)  # TODO: got [nan, nan, nan, nan]
        a = m.sample().numpy()[0]
        return a, prob

    # Calculate loss
    def loss_func(self, state, action, state_value):
        self.train()
        if state.size(dim=0) == 1:  # no squeeze if [1, 4, 84, 84, 1]
            state = state.__array__()
            state = torch.tensor(state)
            state = state.squeeze()
            state = state.unsqueeze(0)
            logits, values = self.forward(state.transpose(1, 3))

        else:  # squeeze if [n, 4, 84, 84, 1]
            state = state.__array__()
            state = torch.tensor(state)
            state = state.squeeze()
            logits, values = self.forward(state.transpose(1, 3))

        # print(state_value.shape)
        # print(values.squeeze().shape)
        td_error = state_value - values.squeeze()
        loss_value = td_error.pow(2)

        prob = F.softmax(logits, dim=1)
        m = self.distribution(prob)
        loss_policy = -(m.log_prob(action) * td_error.detach().squeeze())

        total_loss = (loss_value + loss_policy).mean()
        return total_loss





