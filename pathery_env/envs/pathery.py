from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
  right = 0
  up = 1
  left = 2
  down = 3

class CellType(Enum):
  OPEN = 0
  BLOCKED_PRE_EXISTING = 1
  BLOCKED_PLAYER_PLACED = 2
  START = 3
  GOAL = 4

class PatheryEnv(gym.Env):
  metadata = {"render_modes": ["ansi"], "render_fps": 4}

  def __init__(self, render_mode=None):
    # Initialize grid size
    self.grid_size = (2, 6)

    # Initialize grid with OPEN cells
    self.grid = np.zeros(self.grid_size, dtype=np.int32)

    # Observation space: Each cell type is a discrete value
    self.observation_space = spaces.MultiDiscrete(np.full((self.grid_size[0], self.grid_size[1]), len(CellType)))

    self.window_size = 512  # The size of the PyGame window

    # Possible actions are which 2d position to place a block in
    self.action_space = spaces.MultiDiscrete((self.grid_size[0], self.grid_size[1]))

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

  def _get_obs(self):
    return self.grid

  def _get_info(self):
    return {}

  def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    # Set the number of blocks that the user can place
    self.remainingBlocks = 3

    # Place the start in top left
    self.grid[0][0] = CellType.START.value

    # Place the end in bottom right
    self.grid[self.grid_size[0]-1][self.grid_size[1]-1] = CellType.GOAL.value

    observation = self._get_obs()
    info = self._get_info()

    return observation, info

  def calculateShortestPath(self):
    return 0

  def step(self, action):
    if self.grid[action[0]][action[1]] == CellType.OPEN.value:
      self.grid[action[0]][action[1]] = CellType.BLOCKED_PLAYER_PLACED.value
      self.remainingBlocks -= 1
    else:
      return self._get_obs(), -1, False, False, self._get_info()
    
    terminated = self.remainingBlocks == 0
    reward = self.calculateShortestPath() if terminated else 0
    observation = self._get_obs()
    info = self._get_info()

    return observation, reward, terminated, False, info

  def render(self):
    if self.render_mode == "ansi":
      return self._render_ansi()

  def _render_ansi(self):
    ansi_map = {
      CellType.OPEN: '░',  # Open cells
      CellType.BLOCKED_PRE_EXISTING: '█',  # Blocked by pre-existing
      CellType.BLOCKED_PLAYER_PLACED: '#',  # Blocked by player
      CellType.START: 'S',  # Start
      CellType.GOAL: 'G'   # Goal
    }
    top_border = "+" + "-" * (self.grid_size[1] * 2 - 1) + "+"
    output = top_border + '\n'
    for row in self.grid:
      output += '|' + '|'.join(ansi_map[CellType(val)] for val in row) + '|\n'
    output += top_border + '\n'
    output += f'Remaining blocks: {self.remainingBlocks}'
    return output

  def close(self):
    if self.window is not None:
      pygame.display.quit()
      pygame.quit()
