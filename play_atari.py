# DQN implementation for playing Atari games
# Vincent Zhu

import gym
import numpy as np
import random
from collections import namedtuple, deque
import matplotlib
import matplotlib.pyplot as plt
import stable_baselines3

import torch
from torch import nn
import torchvision

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

import model
from model import DQN
import preprocess
from preprocess import *
from gym.wrappers import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess Atari environment
# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#id2

# env = make_atari_env('BreakoutNoFrameskip-v4')
# env = VecFrameStack(env, n_stack=4)
# print("obs: ", env.observation_space.shape)
# print("act: ", env.action_space)

env = gym.make('BreakoutNoFrameskip-v4').unwrapped

env = AtariPreprocessing(env)
env = FrameStack(env, 4)

# env = SkipFrame(env, skip=4)
# env = GrayScaleObservation(env)
# env = ResizeObservation(env, shape=84)
# env = FrameStack(env, num_stack=4)

print("obs: ", env.observation_space.shape)
print("act: ", env.action_space)


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward', 'done')
)


# Experience replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        return len(self.memory) == self.capacity


# Optimization
def optimize(q_value, action, double_q, target_q, done, reward):
    gamma = 0.99
    loss_fn = nn.SmoothL1Loss(reduction='mean')

    # Calculate Q(s, a; theta_i)
    q_value = q_value.to(device)
    action = action.to(device)
    double_q = double_q.to(device)
    target_q = target_q.to(device)
    done = done.to(device)
    reward = reward.to(device)

    q_sampled_action = q_value.gather(-1, action.to(torch.long).unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        # Get best action: argmax Q(s', a'; theta_i)
        next_action = torch.argmax(double_q, -1)

        # Get q value from target network for best action
        q_next_action = target_q.gather(-1, next_action.unsqueeze(-1)).squeeze(-1)

        # Update q value
        q_update = reward + gamma * q_next_action
        # q_update = reward + gamma * q_next_action * (1 - done)

    # Error
    td_error = q_sampled_action - q_update

    # Loss
    loss = loss_fn(q_sampled_action, q_update)

    # loss = torch.mean(weights * losses)
    return td_error, loss


# Initialize
updates = 1000000
epochs = 10000
update_target_model_every = 250
learning_rate = 1e-4
batch_size = 32
memory = ReplayMemory(1000000)

# Neural network
model = DQN().to(device)
target_model = DQN().to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

# Epsilon-greedy choosing action
exploration_rate = 1.0
exploration_rate_decay = 0.999985
exploration_rate_min = 0.1


def choose_action(q_value):
    global exploration_rate, exploration_rate_decay, exploration_rate_min

    with torch.no_grad():
        greedy_action = torch.argmax(q_value, dim=-1)
        random_action = torch.randint(q_value.shape[-1], greedy_action.shape, device=device)
        # is_choose_rand = exploration_rate < exploration_rate_min
        is_choose_rand = torch.rand(greedy_action.shape, device=device) < exploration_rate
        # print(is_choose_rand, exploration_rate)

        a = torch.where(is_choose_rand, random_action, greedy_action).cpu().numpy()
        return a


# Scale observations from [0, 255] to [0, 1]
def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=device) / 255.


# Sample from experience
def simulate(state):
    with torch.no_grad():
        state = state.__array__()
        if device == "cuda":
            state = torch.tensor(state).cuda()
        else:
            state = torch.tensor(state)
        state = state.unsqueeze(0)
        q_value = model(obs_to_torch(state))
        action = choose_action(q_value)
        next_state, reward, done, info = env.step(action)
        # env.render()

        if device == "cuda":
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            r = torch.tensor([reward]).cuda()
            d = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            r = torch.tensor([reward])
            d = torch.tensor([done])

        memory.push(state, action, next_state, r, d)
        state = next_state
        return state, reward, done

rewards_plot = []

def plot():
    plt.clf()
    episode_rewards_plot = torch.tensor(rewards_plot, dtype=torch.float)
    # average_rewards_plot = torch.tensor(average_rewards, dtype=torch.float)
    # mean_rewards_plot = torch.tensor(mean_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Frames')
    plt.ylabel('Rewards')
    plt.plot(episode_rewards_plot.numpy())
    # plt.plot(average_rewards_plot.numpy())
    # plt.plot(mean_rewards_plot.numpy())
    # Take x episode averages and plot them too
    # if len(episode_rewards_plot) >= 100:
    #     means = episode_rewards_plot.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())
    plt.pause(0.01)


# Training
target_model.load_state_dict(model.state_dict())
obs = env.reset()
sum_reward = 0

for update in range(updates):
    # sum_reward = 0

    next_obs, reward, done = simulate(obs)
    obs = next_obs
    sum_reward += reward

    if done:
        print(update, end=" -- ")
        print("reward: ", sum_reward, end=" -- ")
        print("epsilon: ", exploration_rate)
        obs = env.reset()
        sum_reward = 0
        if len(memory) > 10000:
            rewards_plot.append(sum_reward)
            plot()


    if len(memory) > 10000:
        # Decrease exploration_rate
        exploration_rate = exploration_rate * exploration_rate_decay
        exploration_rate = max(exploration_rate_min, exploration_rate)
        for _ in range(4):
            env.render()
            # Sample from replay memory
            samples = memory.sample(batch_size) # list of batch_size
            state, action, next_state, reward, done = map(torch.stack, zip(*samples))
            # return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

            # Get predicted q-value
            q = model(obs_to_torch(state.squeeze()))

            # Get predicted q-value of the next state for double q-learning
            with torch.no_grad():
                double_q = model(obs_to_torch(next_state))
                target_q = target_model(obs_to_torch(next_state))

            # Calculate error and loss
            td_error, loss = optimize(q,
                                      action.squeeze(),
                                      double_q,
                                      target_q,
                                      done.squeeze(),
                                      reward.squeeze())

            # Zero out calculated gradients
            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()
            # Update parameters based on gradients
            for param in model.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

        # Update target network
        if update % update_target_model_every == 0:
            target_model.load_state_dict(model.state_dict())
