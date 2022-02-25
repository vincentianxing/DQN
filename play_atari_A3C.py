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
# import optim

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


def sync(optimizer, local_net, global_net, done, next_state, buffer_s, buffer_a, buffer_r, gamma):
    if done:
        state_value = 0.  # terminal state
    else:
        state_value = local_net(obs_to_torch(next_state).squeeze().unsqueeze(0))[-1].data.numpy()[0, 0]

    # Store in buffer
    buffer_v_target = []
    # n-step
    for r in buffer_r[::-1]:  # reverse buffer_r
        state_value = r + gamma * state_value
        buffer_v_target.append(state_value)
    buffer_v_target.reverse()

    # Calculate loss
    # print(len(buffer_s), "!!!!!!!!!!")
    loss = local_net.loss_func(
        wrap_to_torch(np.vstack(buffer_s)),
        wrap_to_torch(np.array(buffer_a), dtype=np.int64) if buffer_a[0].dtype == np.int64 else wrap_to_torch(
            np.vstack(buffer_a)),
        wrap_to_torch(np.array(buffer_v_target)[:, None])
    )
    # print(loss)

    # Calculate local gradients and push to global
    optimizer.zero_grad()
    loss.backward()

    # Clip
    torch.nn.utils.clip_grad_norm_(local_net.parameters(), 40)

    for local_param, global_param in zip(local_net.parameters(), global_net.parameters()):
        if global_param.grad is not None:
            return
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
    res_queue.put(ep_r)
    # res_queue.put(global_ep_r.value)
    print(name, ep_r)
    print(
        name,
        "ep: ", global_ep.value,
        " -- ep reward: ", global_ep_r.value
    )


# Multiprocessing
class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.global_ep, self.global_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_net, self.optimizer = global_net, optimizer
        self.local_net = A3C()

        # self.env = gym.make('SpaceInvadersNoFrameskip-v4').unwrapped

        # self.env = gym.make('PongNoFrameskip-v4').unwrapped
        # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

        self.env = gym.make('BreakoutNoFrameskip-v4').unwrapped
        # ['NOOP', 'FIRE', 'RIGHT', 'LEFT']

        # self.env = AtariPreprocessing(self.env)
        # self.env = FrameStack(self.env, 4)

        print(self.env.unwrapped.get_action_meanings())

        self.env = SkipFrame(self.env, skip=4)
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, shape=84)
        self.env = FrameStack(self.env, num_stack=4)

        self.lives = 0

    def run(self):
        total_step = 1
        while self.global_ep.value < 10000:
            state = self.env.reset()
            # print("reset shape: ", state.shape)
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                # if self.name == 'w00':
                self.env.render()

                # Perform action according to policy
                state = state.__array__()
                state = torch.tensor(state)
                state = state.squeeze()
                state = state.unsqueeze(0)
                print(self.name, end=" ")
                action = self.local_net.choose_action(obs_to_torch(state))
                next_state, reward, done, info = self.env.step(action)

                next_state = next_state.__array__()
                next_state = torch.tensor(next_state)
                next_state = next_state.squeeze()
                next_state = next_state.unsqueeze(0)
                next_state = obs_to_torch(next_state)
                # torch.set_printoptions(profile="full")
                # print("compare")
                # print(torch.eq(torch.tensor(state).squeeze(), torch.tensor(next_state).squeeze()))

                # reward = clip_reward(reward)

                reset = done
                new_lives = self.env.ale.lives()
                if new_lives < self.lives:
                    done = True
                self.lives = new_lives

                if reset:
                    print("----------")

                if done:
                    reward = 0

                ep_r += reward

                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)

                if total_step % 10 == 0 or reset:
                    # Sync
                    sync(self.optimizer, self.local_net, self.global_net, reset, next_state, buffer_s, buffer_a,
                         buffer_r, 0.99)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if reset:
                        global_update(self.global_ep, self.global_ep_r, ep_r, self.res_queue, self.name)
                        break

                state = next_state
                total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    global_net = A3C()
    global_net.share_memory()
    # optimizer = torch.optim.Adam(global_net.parameters(), lr=1e-4)
    optimizer = SharedAdam(global_net.parameters(), lr=5e-4)
    # optimizer.share_memory()
    global_ep = mp.Value('i', 0)
    global_ep_r = mp.Value('d', 0.)
    res_queue = mp.Queue()

    # Training in parallel
    workers = [Worker(global_net, optimizer, global_ep, global_ep_r, res_queue, i)
               for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []  # episode reward

    while True:
        r = res_queue.get()
        r_clone = r
        if r is not None:
            res.append(r_clone)
        else:
            break
        plt.clf()
        plt.title('Training...')
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.plot(res)
        plt.pause(0.01)

    [w.join() for w in workers]

    plt.show()
