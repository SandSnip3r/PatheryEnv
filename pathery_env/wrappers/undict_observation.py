import gymnasium as gym
import numpy as np

from pathery_env.envs.pathery import PatheryEnv

class UnDictObservationWrapper(gym.ObservationWrapper):

  def __init__(self, env):
    super().__init__(env)
    self.observation_space = env.observation_space[PatheryEnv.OBSERVATION_BOARD_STR]

  def observation(self, observation):
    return observation[PatheryEnv.OBSERVATION_BOARD_STR]