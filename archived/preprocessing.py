from typing import Optional

import numpy as np
import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit

try:
    import cv2
except ImportError:
    cv2 = None


class Preprocessing(gym.Wrapper):
    r"""Atari 2600 preprocessings.
    This class follows the guidelines in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".
    Specifically:
    * NoopReset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost: turned off by default. Not recommended by Machado et al. (2018).
    * Resize to a square image: 84x84 by default
    * Grayscale observation: optional
    * Scale observation: optional
    Args:
        env (Env): environment
        noop_max (int): max number of no-ops
        frame_skip (int): the frequency at which the agent experiences the game.
        screen_size (int): resize Atari frame
        terminal_on_life_loss (bool): if True, then step() returns done=True whenever a
            life is lost.
        grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
            is returned.
        grayscale_newaxis (bool): if True and grayscale_obs=True, then a channel axis is added to
            grayscale observations to make them 3-dimensional.
        scale_obs (bool): if True, then observation normalized in range [0,1] is returned. It also limits memory
            optimization benefits of FrameStack Wrapper.
    """

    def __init__(
        self,
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    ):
        super().__init__(env)
        assert (
            cv2 is not None
        ), "opencv-python package not installed! Try running pip install gym[atari] to get dependencies  for atari"
        assert frame_skip > 0
        assert screen_size > 0
        assert noop_max >= 0
        if frame_skip > 1:
            assert "NoFrameskip" in env.spec.id, (
                "disable frame-skipping in the original env. for more than one"
                " frame-skip as it will be done by the wrapper"
            )
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale_obs = grayscale_obs
        self.grayscale_newaxis = grayscale_newaxis
        self.scale_obs = scale_obs

        # buffer of most recent two observations for max pooling
        if grayscale_obs:
            self.obs_buffer = [
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
            ]
        else:
            self.obs_buffer = [
                np.empty(env.observation_space.shape, dtype=np.uint8),
                np.empty(env.observation_space.shape, dtype=np.uint8),
            ]

        self.ale = env.unwrapped.ale
        self.lives = 0
        self.game_over = False

        _low, _high, _obs_dtype = (
            (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
        )
        _shape = (screen_size, screen_size, 1 if grayscale_obs else 3)
        if grayscale_obs and not grayscale_newaxis:
            _shape = _shape[:-1]  # Remove channel axis
        self.observation_space = Box(
            low=_low, high=_high, shape=_shape, dtype=_obs_dtype
        )

    def step(env, action):
        R = 0.0

        for t in range(env.frame_skip):
            _, reward, done, info = env.env.step(action)
            R += reward
            env.game_over = done

            if env.terminal_on_life_loss:
                new_lives = env.ale.lives()
                done = done or new_lives < env.lives
                env.lives = new_lives

            if done:
                break
            if t == env.frame_skip - 2:
                if env.grayscale_obs:
                    env.ale.getScreenGrayscale(env.obs_buffer[1])
                else:
                    env.ale.getScreenRGB(env.obs_buffer[1])
            elif t == env.frame_skip - 1:
                if env.grayscale_obs:
                    env.ale.getScreenGrayscale(env.obs_buffer[0])
                else:
                    env.ale.getScreenRGB(env.obs_buffer[0])
        return env.get_obs(), R, done, info

    def reset(self, seed: Optional[int] = None, **kwargs):
        # NoopReset
        self.env.reset(seed=seed, **kwargs)
        noops = (
            self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
            if self.noop_max > 0
            else 0
        )
        for _ in range(noops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset(**kwargs)

        self.lives = self.ale.lives()
        if self.grayscale_obs:
            self.ale.getScreenGrayscale(self.obs_buffer[0])
        else:
            self.ale.getScreenRGB(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)
        return self.get_obs()

    def get_obs(env):
        if env.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(env.obs_buffer[0], env.obs_buffer[1], out=env.obs_buffer[0])
        obs = cv2.resize(
            env.obs_buffer[0],
            (env.screen_size, env.screen_size),
            interpolation=cv2.INTER_AREA,
        )

        if env.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)

        if env.grayscale_obs and env.grayscale_newaxis:
            obs = np.expand_dims(obs, axis=-1)  # Add a channel axis
        return obs