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
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

import DQN_model
from DQN_model import DQN
import preprocess
from preprocess import *
from gym.wrappers import *

import warnings
import gc

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
# env = gym.make('PongNoFrameskip-v4').unwrapped
# env = gym.make('SpaceInvadersNoFrameskip-v4').unwrapped

env = AtariPreprocessing(env)
env = FrameStack(env, 4)
print(env.reset())

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
    # Use L2 loss
    # loss_fn = nn.MSELoss(reduction='mean')

    # Use L1 loss
    # loss_fn = nn.SmoothL1Loss(reduction='mean')

    # Use Huber loss, same as L1 when default delta=1.0
    loss_fn = nn.HuberLoss(reduction='mean')

    # Calculate Q(s, a; theta_i)
    q_value = q_value.to(device)
    action = action.to(device)
    double_q = double_q.to(device)
    target_q = target_q.to(device)
    done = done.to(device)
    reward = reward.to(device)

    # print(q_value)
    q_sampled_action = q_value.gather(-1, action.to(torch.long).unsqueeze(-1)).squeeze(-1)
    # print(q_sampled_action)

    with torch.no_grad():  # done = 1
        # Get best action: argmax Q(s', a'; theta_i)
        next_action = torch.argmax(double_q, -1)

        # Get q value from target network for best action
        q_next_action = target_q.gather(-1, next_action.unsqueeze(-1)).squeeze(-1)

        # Update q value
        q_update = reward + gamma * q_next_action * (1 - done.long())

    # Error
    td_error = q_sampled_action - q_update

    # Loss
    # TODO: .detach().item() only gets value
    loss = loss_fn(q_sampled_action, q_update)

    # loss = torch.mean(weights * losses)
    return td_error, loss


# Initialize
frames = 10000000
random_frames = 10000  # 50000
update_target_model_every = 10000  # 250
batch_size = 32
memory = ReplayMemory(500000)  # 1000000

# Neural network
model = DQN().to(device)
target_model = DQN().to(device)

# Optimizer
learning_rate = 3e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    return torch.tensor(obs, dtype=torch.float32, device=device) / 255


lives = 0


# Sample from experience
def simulate(state):
    global lives
    reset = False
    with torch.no_grad():
        state = state.__array__()
        if device == "cuda":
            state = torch.tensor(state).cuda()
        else:
            state = torch.tensor(state)
        state = state.unsqueeze(0)
        print("simulate shape: ", obs_to_torch(state).shape)
        q_value = model(obs_to_torch(state))
        action = choose_action(q_value)
        next_state, reward, done, info = env.step(action)

        # Passing the terminal flag to the replay memory when a life is lost
        # without resetting the game
        reset = done
        new_lives = env.ale.lives()
        if new_lives < lives:
            done = True
        # print(new_lives, lives, done, reset)
        lives = new_lives

        # Clipping reward
        # reward = clip_reward(reward)
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

        # Store the experience
        memory.push(state, action, next_state, r, d)
        state = next_state
        return state, reward, done, reset


def clip_reward(r):
    if r > 0:
        return 1
    elif r == 0:
        return 0
    else:
        return -1


rewards_plot = []


# Plot with Matplotlib
def plot():
    plt.clf()
    episode_rewards_plot = torch.tensor(rewards_plot, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    plt.plot(episode_rewards_plot.numpy())
    plt.pause(0.01)


# Plot with Tensorboard
writer = SummaryWriter()

# Training
target_model.load_state_dict(model.state_dict())
obs = env.reset()
sum_reward = 0
episode = 0

for frame in range(frames):

    print(obs.shape)
    next_obs, reward, done, reset = simulate(obs)
    # torch.set_printoptions(profile="full")
    print(next_obs.shape)
    obs = next_obs
    sum_reward += reward

    if reset:
        mean_reward = np.mean(rewards_plot[-50:])
        print(frame, end=" -- ")
        print(episode, end=" -- ")
        print("reward: ", sum_reward, end=" -- ")
        print("last 50 mean: ", mean_reward, end=" -- ")
        print("epsilon: ", exploration_rate)
        obs = env.reset()
        if len(memory) > random_frames:
            episode += 1
            rewards_plot.append(sum_reward)
            plot()
            writer.add_scalar("train", sum_reward, frame)
        sum_reward = 0

    if len(memory) > random_frames:

        # Schedule exploration rate
        slope = -(1.0 - exploration_rate_min) / 1000000
        intercept = 1.0 - slope * random_frames
        exploration_rate = slope * frame + intercept
        # exploration_rate = exploration_rate * exploration_rate_decay
        exploration_rate = max(exploration_rate_min, exploration_rate)

        # Optimize
        if frame % 1 == 0:
            env.render()

            # Sample from replay memory
            samples = memory.sample(batch_size)  # list of batch_size
            state, action, next_state, reward, done = map(torch.stack, zip(*samples))

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
        if frame % update_target_model_every == 0:
            target_model.load_state_dict(model.state_dict())

env.close()
writer.flush()
writer.close()
