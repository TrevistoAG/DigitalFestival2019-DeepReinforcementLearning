"""
Environments and wrappers for Sonic training.
"""

import gym
import numpy as np
import retro
import cv2
from gym import spaces
from stable_baselines.common.atari_wrappers import FrameStack

def make_env(game=None, state=None, stack=False, scale_rew=True, allowbacktrace=True, custom=True, render=False):
    """
    Create an environment with some standard wrappers.
    """
    env = retro.RetroEnv(game=game, state=state, scenario="./contest.json")
    env.seed(0)

    env = SonicDiscretizer(env)
    env = WarpFrameRGB(env)
    if scale_rew:
        env = RewardScaler(env)
    if custom:
        env = CustomGym(env, render=render)
    if allowbacktrace:
        env = AllowBacktracking(env, render=render)
    if stack:
        env = FrameStack(env, 4)
    return env


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """

    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [[], ['LEFT'], ['RIGHT'], ['B'], ['RIGHT', 'B'], ['DOWN'], ['DOWN', 'B'], ['RIGHT', 'B'], ['DOWN'],
                   ['B'], []]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """

    def reward(self, reward):
        return reward * 0.01


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """

    def __init__(self, env, render=False):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0
        self.render = render

    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

class CustomGym(gym.Wrapper):
    """
    add custom features
    """

    def __init__(self, env, render=False):
        super(CustomGym, self).__init__(env)
        self.render = render

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.render:
            self.env.render()
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info


class WarpFrameRGB(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 3), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame