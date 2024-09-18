import gymnasium as gym
import numpy as np

from pathery_env.envs.pathery import CellType
from pathery_env.envs.pathery import PatheryEnv

class ActionMaskObservationWrapper(gym.ObservationWrapper):

  OBSERVATION_ACTION_MASK_STR = 'action_mask'

  def __init__(self, env):

    def isWrappedBy(env, wrapper_type):
      """Recursively unwrap env to check if any wrapper is of type wrapper_type."""
      current_env = env
      while isinstance(current_env, gym.Wrapper):
        if isinstance(current_env, wrapper_type):
          return True
        current_env = current_env.env  # Unwrap to the next level
      return False

    super().__init__(env)
    self.observation_space = gym.spaces.Dict({
      PatheryEnv.OBSERVATION_BOARD_STR: env.observation_space[PatheryEnv.OBSERVATION_BOARD_STR],
      ActionMaskObservationWrapper.OBSERVATION_ACTION_MASK_STR: gym.spaces.Box(low=0, high=1, shape=(self.unwrapped.gridSize[0], self.unwrapped.gridSize[1]), dtype=np.int8)
    })

  def step(self, action):
    # For debugging help, check if the action is invalid, based on the grid
    if self.unwrapped.grid[action[0]][action[1]] != CellType.OPEN.value:
      raise ValueError(f'Invalid action {action}')
    return super().step(action)

  def observation(self, observation):
    mask = (observation[PatheryEnv.OBSERVATION_BOARD_STR][CellType.OPEN.value] == 1.0)
    observation[ActionMaskObservationWrapper.OBSERVATION_ACTION_MASK_STR] = mask.astype(np.int8)
    return observation