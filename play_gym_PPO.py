# PPO implementation for gym env
# Vincent Zhu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, set_start_method
from shared_adam import SharedAdam
import numpy as np
import gym
import os
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass

os.environ["OMP_NUM_THREADS"] = "1"
env = gym.make('CartPole-v0')
# print(type(env))
# exit()
input_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.95
CLIP_RANGE = 0.2
VF_COEFF = 0.5  # 1
ENTROPY_COEFF = 0.01

STEPS = 128  # HORIZON
EPOCHS = 4
N_WORKERS = 8
BATCH_SIZE = N_WORKERS * STEPS
MINI_BATCH_SIZE = BATCH_SIZE // 8  # <= 8 * 128
LEARNING_RATE = 1e-3
UPDATES = 10000


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 64)
        self.pi2 = nn.Linear(64, a_dim)
        self.v1 = nn.Linear(s_dim, 64)
        self.v2 = nn.Linear(64, 1)
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
        pi = self.distribution(prob)
        return pi.sample().numpy()[0]

    def loss_func(self, state, action, state_value):
        self.train()
        logits, values = self.forward(state)
        advantages = state_value - values
        value_loss = advantages.pow(2)

        prob = F.softmax(logits, dim=1)
        pi = self.distribution(prob)
        exp_v = pi.log_prob(action) * advantages.detach().squeeze()
        policy_loss = -exp_v
        loss = (value_loss + policy_loss).mean()
        return loss

    def normalize_advantage(self, adv):
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def loss_ppo(self, batch):
        self.train()
        logits, values = self.forward(batch['obs'])
        prob = F.softmax(logits)
        # print(prob)
        pi = self.distribution(prob)

        log_pis = pi.log_prob(batch['actions'])
        advantages = batch['advantages']
        advantages_normalized = self.normalize_advantage(advantages)

        # Calculate L_clip for policy loss
        ratio = torch.exp(log_pis - batch['log_pis'])
        ratio_clipped = ratio.clamp(min=1.0 - CLIP_RANGE,
                                    max=1.0 + CLIP_RANGE)
        policy_loss = torch.min(ratio * advantages,
                                ratio_clipped * advantages)
        policy_loss = policy_loss.mean()

        # Calculate entropy bonus
        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        # Calculate L_vf for value loss
        value_loss = (batch['values'] - values).pow(2)
        value_loss = value_loss.mean()

        # sampled_return = batch['values'] + batch['advantages']
        # clipped_value = batch['values'] + (values - batch['values']).clamp(min=-CLIP_RANGE,
        #                                                                    max=CLIP_RANGE)
        # vf_loss = torch.max((values - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        # value_loss = 0.5 * vf_loss.mean()

        # Calculate total loss = L_clip + L_vf + L_entropy
        loss = -(policy_loss - VF_COEFF * value_loss + ENTROPY_COEFF * entropy_bonus)

        return loss


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def train(optimizer, local_net, global_net, samples):
    # Training based on samples
    for _ in range(1):
        # shuffle for each epoch
        indexes = torch.randperm(STEPS) # BATCH_SIZE
        for start in range(0, STEPS, MINI_BATCH_SIZE):
            # get mini batch
            end = start + MINI_BATCH_SIZE
            mini_batch_indexes = indexes[start: end]
            mini_batch = {}
            for k, v in samples.items():
                for i in mini_batch_indexes:
                    mini_batch[k] = v[mini_batch_indexes[i]]

            # train
            loss = local_net.loss_ppo(mini_batch)

            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_net.parameters(), max_norm=0.5)
            for local_param, global_param in zip(local_net.parameters(), global_net.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()

        # Pull global parameters
        local_net.load_state_dict(global_net.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
        "| ", ep_r
    )


class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_net, self.optimizer = global_net, optimizer
        self.local_net = Net(input_size, n_actions)
        self.env = gym.make('CartPole-v0').unwrapped

    def calculate_advantages(self, next_state, buffer_r, buffer_done, buffer_v):
        adv_buffer = []
        last_adv = 0
        _, last_values = self.local_net.forward(v_wrap(next_state[None, :]))

        for t in reversed(range(STEPS)):
            # terminal state mask
            mask = 1.0 - buffer_done[t]
            last_values = last_values * mask
            last_adv = last_adv * mask

            delta = buffer_r[t] + GAMMA * last_values - buffer_v[t]
            last_adv = delta + GAMMA * LAMBDA * last_adv
            adv_buffer.append(last_adv)
            last_values = buffer_v[t]

        # TODO: check adv_buffer.reverse()
        adv_buffer.reverse()
        return adv_buffer

    def run(self):
        while self.g_ep.value < UPDATES:

            # Sampling
            buffer_s = []
            buffer_a = []
            buffer_r = []
            buffer_done = []
            buffer_log_pi = []
            buffer_v = []
            buffer_adv = []
            state = self.env.reset()
            ep_r = 0.

            # Run old policy for STEPS
            for t in range(STEPS):
                with torch.no_grad():
                    # Choose action
                    action = self.local_net.choose_action(v_wrap(state[None, :]))

                    # Step with old policy
                    next_state, reward, done, info = self.env.step(action)
                    # if done:
                    #     reward = -1

                    # Calculate log_pi
                    logits, values = self.local_net.forward(v_wrap(state[None, :]))
                    prob = F.softmax(logits, dim=1)
                    pi = self.local_net.distribution(prob)
                    a = pi.sample()
                    log_pi = pi.log_prob(a).detach()

                    # Add to buffer
                    buffer_a.append(action)
                    buffer_s.append(state)
                    buffer_r.append(reward)
                    buffer_done.append(done)
                    buffer_log_pi.append(log_pi)
                    buffer_v.append(values)
                    ep_r += reward

                    state = next_state

            # Calculate advantage estimates
            buffer_adv = self.calculate_advantages(next_state, buffer_r, buffer_done, buffer_v)

            samples = {
                'obs': v_wrap(np.vstack(buffer_s)),
                'actions': v_wrap(np.array(buffer_a), dtype=np.int64) if buffer_a[0].dtype == np.int64 else v_wrap(
                    np.vstack(buffer_a)),
                'values': v_wrap(np.array(buffer_v)[:, None]),
                'log_pis': v_wrap(np.array(buffer_log_pi)),
                'advantages': buffer_adv
            }

            # Training
            train(self.optimizer, self.local_net, self.global_net, samples)

            if done:  # done and print information
                record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)

        self.res_queue.put(None)


if __name__ == "__main__":
    global_net = Net(input_size, n_actions)  # global network
    global_net.share_memory()  # share the global parameters in multiprocessing
    optimizer = SharedAdam(global_net.parameters(), lr=LEARNING_RATE)  # global optimizer
    global_ep = mp.Value('i', 0)
    global_ep_r = mp.Value('d', 0.)
    res_queue = mp.Queue()

    # parallel training
    workers = [Worker(global_net, optimizer, global_ep, global_ep_r, res_queue, i) for i in range(N_WORKERS)]
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
