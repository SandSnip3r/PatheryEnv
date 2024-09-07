import gymnasium as gym
import numpy as np

class FlattenActionWrapper(gym.ActionWrapper):
  def __init__(self, env):
    super().__init__(env)
    self.original_action_space = env.action_space
    # Note: This assumes that the original environment's action space is a MultiDiscrete
    self.action_space = gym.spaces.Discrete(np.prod(env.action_space.nvec))
  
  def action(self, action):
    # Convert flattened action back to the original action space
    return np.unravel_index(action, self.original_action_space.nvec)
  