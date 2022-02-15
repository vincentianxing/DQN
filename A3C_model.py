# A3C neural network architecture
# Vincent Zhu

import torch
from torch import nn


class A3C(nn.Module):
    def __init__(self, input_size=4, n_actions=6):
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
        action_value = self.policy(x)
        state_value = self.value(x)

        return action_value, state_value




