import gym
from gym import envs
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# For storing the experience of agent
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
# Neural network
import torch.nn as nn
# Optimization
import torch.optim as optim
import torch.nn.functional as F
# Vision tasks
import torchvision.transforms as T

# check available env
# print(envs.registry.all())

# get pendulum env
env = gym.make('Pendulum-v1').unwrapped

# env.action_space, env.observation_space of type Space
# env.observation_space.high/low to check bounds
# Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
print("obs_space: ", env.observation_space)
print("act_space: ", env.action_space)  # Box([-2.], [2.], (1,), float32)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# represent a single transition from state to next_state
Transition = namedtuple('Transition', 'state',
                        'action', 'next_state', 'reward')


# Replay memory storing transition observed by agent
# sampled to get a decorrelated batch
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # save a transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

        # step(action) returns:
        # observation(object), reward(float), done(boolean), info(dict)
        # on each step, agent choose action, env returns obs and reward
