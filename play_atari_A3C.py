# A3C implementation for playing Atari games
# Vincent Zhu

import gym
import numpy as np
import random
from collections import namedtuple, deque
import matplotlib
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, set_start_method

import DQN_model
from A3C_model import A3C
import preprocess
from preprocess import *
from gym.wrappers import *

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
     set_start_method('spawn')
except RuntimeError:
    pass

# Get environment
# env = gym.make('PongNoFrameskip-v4').unwrapped
# env = AtariPreprocessing(env)
# env = FrameStack(env, 4)

# Transition tuple
Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward', 'done')
)

# Global values
GAMMA = 0.99

# Scale observations from [0, 255] to [0, 1]
def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device='cpu') / 255


def wrap_to_torch(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def clip_reward(r):
    if r > 0:
        return 1
    elif r == 0:
        return 0
    else:
        return -1

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

# Choose action based on model
def choose_action(model, state):
    model.eval()
    logits, values = model(state)
    prob = F.softmax(logits, dim=1).data
    m = model.distribution(prob)
    return m.sample().numpy()[0]


# Calculate loss
def loss_func(model, state, action, state_value):
    model.train()
    logits, values = model(state.squeeze())
    td_error = state_value - values

    loss_value = td_error.pow(2)

    prob = F.softmax(logits, dim=1)
    m = model.distribution(prob)
    loss_policy = -(m.log_prob(action) * td_error.detach().squeeze())

    total_loss = (loss_value + loss_policy).mean()
    return total_loss


def sync(optimizer, local_net, global_net, done, next_state, buffer_s, buffer_a, buffer_r, gamma):
    if done:
        state_value = 0.  # terminal state
    else:
        state_value = local_net(obs_to_torch(next_state).squeeze().unsqueeze(0))[-1].data.numpy()[0, 0]

    # Store R in buffer
    buffer_v_target = []
    # n-step
    for r in buffer_r[::-1]:  # reverse buffer_r
        state_value = r + gamma * state_value
        buffer_v_target.append(state_value)
    buffer_v_target.reverse()

    # Calculate loss
    print(len(buffer_s))
    loss = loss_func(
        local_net,
        wrap_to_torch(np.array(buffer_s)), # vstack
        wrap_to_torch(np.array(buffer_a), dtype=np.int64) if buffer_a[0].dtype == np.int64 else wrap_to_torch(
            np.vstack(buffer_a)),
        wrap_to_torch(np.array(buffer_v_target)[:, None])
    )

    # Calculate local gradients and push to global
    optimizer.zero_grad()
    loss.backward()
    for local_param, global_param in zip(local_net.parameters(), global_net.parameters()):
        global_param._grad = local_param.grad
    optimizer.step()

    # Pull global parameters
    local_net.load_state_dict(global_net.state_dict())


def global_update(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "ep: ", global_ep.value,
        " -- ep reward: %.0f", global_ep_r.value
    )


# Multiprocessing
class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.global_ep, self.global_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_net, self.optimizer = global_net, optimizer
        # self.local_net = A3C().to(device)
        self.local_net = A3C()
        self.env = gym.make('PongNoFrameskip-v4').unwrapped
        # self.env = AtariPreprocessing(self.env)
        # self.env = FrameStack(self.env, 4)

        self.env = SkipFrame(self.env, skip=4)
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, shape=84)
        self.env = FrameStack(self.env, num_stack=4)

    def run(self):
        print(self.name, 'running')
        total_step = 1
        while self.global_ep.value < 3000:
            state = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                print(self.name, 'loop2')
                if self.name == 'w00':
                    self.env.render()

                # Perform action according to policy
                action = choose_action(self.local_net, obs_to_torch(state).squeeze().unsqueeze(0))
                next_state, reward, done, info = self.env.step(action)
                reward = clip_reward(reward)

                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)

                if total_step % 5 == 0 or done:
                    # Sync
                    print(self.name, "sync")
                    sync(self.optimizer, self.local_net, self.global_net, done, next_state, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    print(self.name, "sync done")

                    if done:
                        print(self.name, "update")
                        global_update(self.global_ep, self.global_ep_r, ep_r, self.res_queue, self.name)
                        break

                state = next_state
                total_step += 1

        self.res_queue.put(None)


def process_queue(q, l):
    while True:
        print("before")
        r = q.get()
        print("after")
        if r is not None:
            l.append(r)
        else:
            break

if __name__ == "__main__":
    # global_net = A3C().to(device)
    global_net = A3C()
    global_net.share_memory()
    # optimizer = torch.optim.Adam(global_net.parameters(), lr=1e-5)
    optimizer = SharedAdam(global_net.parameters())
    global_ep = mp.Value('i', 0)
    global_ep_r = mp.Value('d', 0.)
    res_queue = mp.Queue()

    # Training in parallel
    workers = [Worker(global_net, optimizer, global_ep, global_ep_r, res_queue, i)
               for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []  # episode reward

    # print(res_queue)

    # while True:
    #     r = res_queue.get()
    #     if r is not None:
    #         res.append(r)
    #     else:
    #         break

    [w.join() for w in workers]

    # plt.plot(res)
    # plt.ylabel('Moving average ep reward')
    # plt.xlabel('Step')
    # plt.show()
