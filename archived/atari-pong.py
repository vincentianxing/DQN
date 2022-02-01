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
import copy

# Pytorch
import torch
# Neural network
import torch.nn as nn
from torch.nn.modules import conv
# Optimization
import torch.optim as optim
import torch.nn.functional as F
# Vision tasks
import torchvision.transforms as T

# Wrappers
from gym.spaces import Box
from gym.wrappers import *
from gym.wrappers import AtariPreprocessing
import cv2

# check available env
# print(envs.registry.all())

# Get env
env = gym.make('BreakoutNoFrameskip-v4').unwrapped
print("Original env:")
print("obs: ", env.observation_space.shape)
print("act: ", env.action_space)

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# represent a single transition from state to next_state
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Define neural network
class DQN(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        print(input_shape, input_shape[0])

        self.online = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def _get_conv_out(self, shape):
        o = self.online(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, model):
        x = x.float().to(device)

        if model == "online":
            online_out = self.online(x).view(x.size()[0], -1)
            return self.fc(online_out)
        elif model == "target":
            target_out = self.target(x).view(x.size()[0], -1)
            return self.fc(target_out)

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


# Apply Wrappers to environment
# env = SkipFrame(env, skip=4)
# env = GrayScaleObservation(env)
# env = ResizeObservation(env, shape=84)
# env = FrameStack(env, num_stack=4)

print("\nPreprocessed env: ")
env = AtariPreprocessing(env)
env = FrameStack(env, 4)
print("obs: ", env.observation_space.shape)
print("act: ", env.action_space)
print()
print(DQN(env.observation_space.shape, env.action_space.n).to(device))

# Initialize network
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

net = DQN(env.observation_space.shape, env.action_space.n).float().to(device)

exploration_rate = 1.0
exploration_rate_decay = 0.999985
exploration_rate_min = 0.1
curr_step = 0


def select_action(state):
    global exploration_rate, exploration_rate_decay, exploration_rate_min, curr_step

    # Exploit with epsilon greedy
    r = np.random.rand()
    if r < exploration_rate:
        action_idx = np.random.randint(env.action_space.n)
        # print("EXPLORE", r, exploration_rate)

    else:
        # with torch.no_grad():
        state = state.__array__()
        if device == "cuda":
            state = torch.tensor(state).cuda()
        else:
            state = torch.tensor(state)
        # print("before: ", state.shape)
        state = state.unsqueeze(0)
        # print("after: ", state.shape)
        action_values = net(state, model="online")
        action_idx = torch.argmax(action_values, axis=1).item()
        # print("EXPLOIT", r, exploration_rate, action_idx)

    # decrease exploration_rate
    exploration_rate = exploration_rate * exploration_rate_decay
    exploration_rate = max(exploration_rate_min, exploration_rate)

    # increment step
    curr_step += 1
    return action_idx


# Replay Memory
memory = deque(maxlen=1000000)
batch_size = 32


# Store the experience to memory (replay buffer)
def cache(state, next_state, action, reward, done):
    # state (LazyFrame),
    # next_state (LazyFrame),
    # action (int),
    # reward (float),
    # done(bool))

    state = state.__array__()
    next_state = next_state.__array__()

    # Convert to tensor
    if device == "cuda":
        state = torch.tensor(state).cuda()
        next_state = torch.tensor(next_state).cuda()
        action = torch.tensor([action]).cuda()
        reward = torch.tensor([reward]).cuda()
        done = torch.tensor([done]).cuda()
    else:
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

    memory.append((state, next_state, action, reward, done,))


# Retrieve a batch of experiences from memory
def recall():
    batch = random.sample(memory, batch_size)
    # print("recall!!!")
    # print(batch)
    # print(len(batch))
    # exit()
    state, next_state, action, reward, done = map(torch.stack, zip(*batch))
    print("recall ", state.shape)
    exit()
    return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


gamma = 0.99


def td_estimate(state, action):
    current_Q = net(state, model="online")[
        np.arange(0, batch_size), action
    ]  # Q_online(s,a)
    return current_Q


@torch.no_grad()
def td_target(reward, next_state, done):
    next_state_Q = net(next_state, model="online")
    best_action = torch.argmax(next_state_Q, axis=1)
    next_Q = net(next_state, model="target")[
        np.arange(0, batch_size), best_action
    ]
    return (reward + (1 - done.float().to(device)) * gamma * next_Q).float().to(device)


# Update with theta
optimizer = torch.optim.Adam(net.parameters(), lr=0.00025)
loss_fn = torch.nn.SmoothL1Loss()


def update_Q_online(td_estimate, td_target):
    loss = loss_fn(td_estimate, td_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def sync_Q_target():
    net.target.load_state_dict(net.online.state_dict())


burnin = 1e4  # min. experiences before training
learn_every = 3  # no. of experiences between updates to Q_online
sync_every = 1e4  # no. of experiences between Q_target & Q_online sync


def learn():
    # sync_Q_target()
    if curr_step % sync_every == 0:
        sync_Q_target()

    # if curr_step % save_every == 0:
    #     save()

    if curr_step < burnin:
        return None, None

    if curr_step % learn_every != 0:
        return None, None

    # Sample from memory
    print("sample")
    state, next_state, action, reward, done = recall()
    state = state.to(device)
    next_state = next_state.to(device)
    action = action.to(device)
    reward = reward.to(device)
    done = done.to(device)

    # Get TD Estimate
    td_est = td_estimate(state, action)

    # Get TD Target
    td_tgt = td_target(reward, next_state, done)

    # Backpropagate loss through Q_online
    loss = update_Q_online(td_est, td_tgt)

    return (td_est.mean().item(), loss)


episode_rewards = []
average_rewards = []
mean_rewards = []


def plot ():
    plt.figure(2)
    plt.clf()
    episode_rewards_plot = torch.tensor(episode_rewards, dtype=torch.float)
    average_rewards_plot = torch.tensor(average_rewards, dtype=torch.float)
    mean_rewards_plot = torch.tensor(mean_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(episode_rewards_plot.numpy())
    plt.plot(average_rewards_plot.numpy())
    plt.plot(mean_rewards_plot.numpy())
    # Take x episode averages and plot them too
    # if len(episode_rewards_plot) >= 100:
    #     means = episode_rewards_plot.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# Training
num_episodes = 40000
total_rewards = 0
for i_episode in range(num_episodes):
    # action = 0
    sum_reward = 0
    state = env.reset()
    # print("reset: ", state)
    # last_screen = get_screen()
    # current_screen = get_screen()
    # state = current_screen - last_screen
    while True:

        # TODO: state is the observation, which is a 4-obs stack, cannot be passed into select_action() and regular step()
        # TODO: AtariPreprocessing.step() "attempted to get missing private attribute '_get_obs'"

        # env.render()

        # Run agent on state
        action = select_action(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

        sum_reward += reward
        # mean_reward = np.mean(total_rewards[-100:])
        # print(done, reward)
        # reward = torch.tensor([reward], device=device)

        # Remember
        # Store the transition in memory
        # memory.push(state, action, next_state, reward)
        cache(state, next_state, action, reward, done)

        # Sample and learn
        learn()

        # Move to the next state
        state = next_state

        if done:
            total_rewards += sum_reward
            average = total_rewards / (i_episode + 1)
            episode_rewards.append(sum_reward)
            average_rewards.append(average)
            mean_rewards.append(np.mean(episode_rewards[-100:]))
            print(sum_reward, np.mean(episode_rewards[-100:]), exploration_rate)
            plot()
            break

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
