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
from torch.nn.modules import conv
# Optimization
import torch.optim as optim
import torch.nn.functional as F
# Vision tasks
import torchvision.transforms as T

from gym.wrappers import *
import baseline_wrappers

# Atari roms
# import ale_py.roms as roms
# print(roms.__all__)

# check available env
# print(envs.registry.all())

# get env
env = gym.make('PongNoFrameskip-v4').unwrapped
print("Original env:")
print("obs: ", env.observation_space.shape)
print("act: ", env.action_space)

# env.action_space, env.observation_space of type Space
# env.observation_space.high/low to check bounds

# print("obs_space: ", env.observation_space)
# print("act_space: ", env.action_space)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# represent a single transition from state to next_state
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Replay memory, storing transition observed by agent, sampled to get a decorrelated batch
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


# Define model Q-network
class DQN(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # print(input_shape, input_shape[0])

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
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

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)


# Test wrappers and network structure
# env = baseline_wrappers.MaxAndSkipEnv(env)
# env = baseline_wrappers.FireResetEnv(env)
# env = baseline_wrappers.ProcessFrame84(env)
# env = baseline_wrappers.ImageToPyTorch(env)
# env = baseline_wrappers.BufferWrapper(env, 4)
# print("obs: ", env.observation_space.shape)
# print("act: ", env.action_space)
# print()
# print(DQN(env.observation_space.shape, env.action_space.n).to(device))

# env2 = gym.make('PongNoFrameskip-v4').unwrapped
# print("obs: ", env2.observation_space)
# print("act: ", env2.action_space)
# print()

print("\nPreprocessed env: ")
env = AtariPreprocessing(env)
env = FrameStack(env, 4)
print("obs: ", env.observation_space.shape)
print("act: ", env.action_space)
print()
print(DQN(env.observation_space.shape, env.action_space.n).to(device))


#     self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
#     self.bn1 = nn.BatchNorm2d(32)
#     self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#     self.bn2 = nn.BatchNorm2d(64)
#     self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#     self.bn3 = nn.BatchNorm2d(64)
#
#     conv_out_size = self.get_conv_out(input_shape)
#
#     self.conv_out = nn.Linear(conv_out_size, 512)
#     self.out = nn.Linear(512, n_actions)
#
#
# def get_conv_out(self, shape):
#     o = self.conv(torch.zeros(1, *shape))
#     return int(np.prod(o.size()))

# Number of Linear input connections depends on output of conv2d layers
# def conv2d_size_out(size, kernel_size=3, stride=1):
#     return (size - (kernel_size - 1) - 1) // stride + 1
# convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
# convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
# linear_input_size = convw * convh * 64
# print(linear_input_size, outputs)
# self.head = nn.Linear(linear_input_size, outputs)

# def forward(self, x):
#     x = x.to(device)
#     x = F.relu(self.bn1(self.conv1(x)))
#     x = F.relu(self.bn2(self.conv2(x)))
#     x = F.relu(self.bn3(self.conv3(x)))
#     x = F.relu(self.conv_out(x))
#     x = self.out(x)
#     return self.head(x.view(x.size(0), -1))


# Input extraction
# compose several transforms of image
# resize = T.Compose([T.ToPILImage(),  # Convert a tensor or an ndarray to PIL Image.
#                     T.Resize(40, interpolation=Image.CUBIC),
#                     T.ToTensor()])


def get_screen():
    # # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # # such as 800x1200x3. Transpose it into torch order (CHW).
    # screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # # Cart is in the lower half, so strip off the top and bottom of the screen
    # _, screen_height, screen_width = screen.shape
    # # Convert to float, rescale, convert to torch tensor
    # # (this doesn't require a copy)
    # screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    # screen = torch.from_numpy(screen)
    # # Resize, and add a batch dimension (BCHW)
    # return resize(screen).unsqueeze(0)
    return AtariPreprocessing._get_obs(env)


# env.reset()
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# init_screen = get_screen()
# _, _, screen_height, screen_width = init_screen.shape

# preprocessing
# env = AtariPreprocessing(env)
# env = FrameStack(env, 4)

n_states = env.observation_space.shape[0]
n_actions = n_actions = env.action_space.n

# frozen network?
policy_net = DQN(env.observation_space.shape, n_actions).to(device)
target_net = DQN(env.observation_space.shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# only update policy network parameters
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

episode_durations = []
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # print(policy_net(state))
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.random()]], device=device, dtype=torch.long)


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# Training
# env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video', force=True)

num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        print(done)
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()