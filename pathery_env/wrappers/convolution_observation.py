import gymnasium as gym
import numpy as np

from pathery_env.envs.pathery import PatheryEnv

class ConvolutionObservationWrapper(gym.ObservationWrapper):

  def __init__(self, env):
    super().__init__(env)
    self.original_board_observation_space = env.observation_space[PatheryEnv.OBSERVATION_BOARD_STR]
    self.channel_count = self.original_board_observation_space[0][0].n
    self.height = self.original_board_observation_space.shape[0]
    self.width = self.original_board_observation_space.shape[1]
    self.observation_space = gym.spaces.Dict({
      **{key: value for key, value in env.observation_space.spaces.items()},
      PatheryEnv.OBSERVATION_BOARD_STR: gym.spaces.Box(low=0.0, high=1.1, shape=(self.channel_count, self.height, self.width), dtype=np.float32)
    })

  def observation(self, observation):
    board = observation[PatheryEnv.OBSERVATION_BOARD_STR]
    oneHot = np.zeros((self.channel_count, self.height, self.width), dtype=np.float32)
    for i in range(self.channel_count):
      oneHot[i] = (board == i)
    return {
      **{key: value for key, value in observation.items()},
      PatheryEnv.OBSERVATION_BOARD_STR: oneHot
    }