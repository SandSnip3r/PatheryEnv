import gymnasium as gym
import numpy as np

from pathery_env.envs.pathery import PatheryEnv

class FlattenBoardObservationWrapper(gym.ObservationWrapper):

  def __init__(self, env):
    super().__init__(env)
    self.originalBoardObservationSpace = env.observation_space[PatheryEnv.OBSERVATION_BOARD_STR]
    newBoardObservationSpace = gym.spaces.utils.flatten_space(self.originalBoardObservationSpace)
    self.observation_space = gym.spaces.Dict({
      **{key: value for key, value in env.observation_space.spaces.items()},
      PatheryEnv.OBSERVATION_BOARD_STR: newBoardObservationSpace
    })

  def observation(self, observation):
    unflattenedBoard = observation[PatheryEnv.OBSERVATION_BOARD_STR]
    observation[PatheryEnv.OBSERVATION_BOARD_STR] = gym.spaces.utils.flatten(self.originalBoardObservationSpace, unflattenedBoard)
    return observation