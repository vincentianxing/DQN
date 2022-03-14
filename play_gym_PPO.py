# A3C implementation for gym env
# Vincent Zhu

import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, set_start_method
from shared_adam import SharedAdam
import gym
import os
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass

os.environ["OMP_NUM_THREADS"] = "1"


env = gym.make('CartPole-v0')
input_size = env.observation_space.shape[0]
n_actions = env.action_space.n


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, state):
        self.eval()
        logits, values = self.forward(state)
        prob = F.softmax(logits, dim=1).data
        # print(prob)
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, state, action, state_value):
        self.train()
        logits, values = self.forward(state)
        advantage = state_value - values
        value_loss = advantage.pow(2)

        prob = F.softmax(logits, dim=1)
        m = self.distribution(prob)
        exp_v = m.log_prob(action) * advantage.detach().squeeze()
        policy_loss = -exp_v
        total_loss = (value_loss + policy_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_net, self.optimizer = global_net, optimizer
        self.local_net = Net(input_size, n_actions)  # local network
        self.env = gym.make('CartPole-v0').unwrapped
        # print(self.env.unwrapped.get_action_meanings())

    def run(self):
        total_step = 1
        while self.g_ep.value < 2500:
            state = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.

            while True:
                # if self.name == 'w00':
                #     self.env.render()

                action = self.local_net.choose_action(v_wrap(state[None, :]))

                next_state, reward, done, info = self.env.step(action)
                if done: reward = -1

                ep_r += reward
                buffer_a.append(action)
                buffer_s.append(state)
                buffer_r.append(reward)

                if total_step % 5 == 0 or done:
                    # sync
                    sync(self.optimizer, self.local_net, self.global_net, done, next_state, buffer_s, buffer_a, buffer_r, 0.9)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break

                state = next_state
                total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    global_net = Net(input_size,n_actions)  # global network
    global_net.share_memory()  # share the global parameters in multiprocessing
    optimizer = SharedAdam(global_net.parameters(), lr=1e-4)  # global optimizer
    global_ep = mp.Value('i', 0)
    global_ep_r = mp.Value('d', 0.)
    res_queue = mp.Queue()

    # parallel training
    workers = [Worker(global_net, optimizer, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []  # record episode reward

    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    [w.join() for w in workers]

    plt.clf()
    plt.title('Training...')
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.plot(res)

    plt.show()
