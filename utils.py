"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def sync(optimizer, local_net, global_net, done, next_state, buffer_s, buffer_a, buffer_r, gamma):
    if done:
        state_value = 0.  # terminal state
    else:
        state_value = local_net.forward(v_wrap(next_state[None, :]))[-1].data.numpy()[0, 0]

    # Store in buffer
    buffer_v_target = []
    for r in buffer_r[::-1]:  # reverse buffer r
        state_value = r + gamma * state_value
        buffer_v_target.append(state_value)
    buffer_v_target.reverse()

    loss = local_net.loss_func(
        v_wrap(np.vstack(buffer_s)),
        v_wrap(np.array(buffer_a), dtype=np.int64) if buffer_a[0].dtype == np.int64 else v_wrap(np.vstack(buffer_a)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(local_net.parameters(), 40)

    for local_param, global_param in zip(local_net.parameters(), global_net.parameters()):
        global_param._grad = local_param.grad
    optimizer.step()

    # pull global parameters
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
    )