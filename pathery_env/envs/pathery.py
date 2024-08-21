from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from collections import deque

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
    self.gridSize = (2, 6)

    # Initialize grid with OPEN cells
    self.grid = np.zeros(self.gridSize, dtype=np.int32)

    # Observation space: Each cell type is a discrete value
    self.observation_space = spaces.MultiDiscrete(np.full((self.gridSize[0], self.gridSize[1]), len(CellType)))

    # Possible actions are which 2d position to place a block in
    self.action_space = spaces.MultiDiscrete((self.gridSize[0], self.gridSize[1]))

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

    self.startPos = (0,0)
    self.goalPos = (self.gridSize[0]-1, self.gridSize[1]-1)

    # Place the start in top left
    self.grid[self.startPos[0]][self.startPos[1]] = CellType.START.value

    # Place the end in bottom right
    self.grid[self.goalPos[0]][self.goalPos[1]] = CellType.GOAL.value

    observation = self._get_obs()
    info = self._get_info()

    return observation, info

  def calculateShortestPath(self):
    # Directions for moving: right, left, down, up
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # Create a queue for BFS and add the starting point
    queue = deque([(self.startPos, 0)])
    
    # Set of visited nodes
    visited = set()
    visited.add(self.startPos)
    
    while queue:
      # Get the current position and the path length to it
      (current, pathLength) = queue.popleft()
      
      # If the current position is the goal, return the path
      if current == self.goalPos:
        return pathLength
      
      # Explore all the possible directions
      for direction in directions:
        # Calculate the next position
        next_position = (current[0] + direction[0], current[1] + direction[1])
        
        # Check if the next position is within the grid bounds
        if (0 <= next_position[0] < self.gridSize[0]) and (0 <= next_position[1] < self.gridSize[1]):
          # Check if the next position is not an obstacle and not visited
          if self.grid[next_position[0]][next_position[1]] in [CellType.OPEN.value, CellType.START.value, CellType.GOAL.value] and next_position not in visited:
            # Add the next position to the queue and mark it as visited
            queue.append((next_position, pathLength+1))
            visited.add(next_position)
    
    # There is no path to the goal
    return 0


  def step(self, action):
    if self.grid[action[0]][action[1]] == CellType.OPEN.value:
      self.grid[action[0]][action[1]] = CellType.BLOCKED_PLAYER_PLACED.value
      self.remainingBlocks -= 1
    else:
      return self._get_obs(), -1, False, False, self._get_info()
    
    pathLength = self.calculateShortestPath()
    
    terminated = (self.remainingBlocks == 0 or pathLength == 0)
    reward = (-1 if pathLength == 0 else pathLength) if terminated else 0
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
    top_border = "+" + "-" * (self.gridSize[1] * 2 - 1) + "+"
    output = top_border + '\n'
    for row in self.grid:
      output += '|' + '|'.join(ansi_map[CellType(val)] for val in row) + '|\n'
    output += top_border + '\n'
    output += f'Remaining blocks: {self.remainingBlocks}'
    return output

  def close(self):
    pass
