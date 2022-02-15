# DQN neural network architecture
# Vincent Zhu

import torch
from torch import nn


class DQN(nn.Module):
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

        # Initialize weights
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)

        # self.conv.apply(init_weights)

        # Final hidden layer: fully-connected and consists of 512 rectifier units.
        # Input: 7 * 7
        self.final_layer = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.final_activation = nn.ReLU()

        # Output layer: a fully-connected linear layer with a single output for each valid action.
        self.head = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_actions),
        )

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
        q = self.head(x)

        return q




