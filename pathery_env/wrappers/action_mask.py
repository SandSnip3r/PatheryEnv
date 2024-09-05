import gymnasium as gym
import numpy as np

from pathery_env.envs.pathery import InternalCellType
from pathery_env.envs.pathery import PatheryEnv

class ActionMaskWrapper(gym.ObservationWrapper):

  OBSERVATION_ACTION_MASK_STR = 'action_mask'

  def __init__(self, env):
    super().__init__(env)
    self.observation_space = gym.spaces.Dict({
      PatheryEnv.OBSERVATION_BOARD_STR: env.observation_space[PatheryEnv.OBSERVATION_BOARD_STR],
      ActionMaskWrapper.OBSERVATION_ACTION_MASK_STR: gym.spaces.Box(low=0, high=1, shape=(self.unwrapped.gridSize[0], self.unwrapped.gridSize[1]), dtype=np.int8)
    })

  def step(self, action):
    # For debugging help, check if the action is invalid, based on the grid
    if self.unwrapped.grid[action[0]][action[1]] != InternalCellType.OPEN.value:
      raise ValueError(f'Invalid action {action}')
    return super().step(action)

  def observation(self, observation):
    mask = (observation[PatheryEnv.OBSERVATION_BOARD_STR] == InternalCellType.OPEN.value)
    observation[ActionMaskWrapper.OBSERVATION_ACTION_MASK_STR] = mask.astype(np.int8)
    return observation