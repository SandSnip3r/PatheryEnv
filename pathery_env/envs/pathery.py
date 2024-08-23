from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from collections import deque

class InternalCellType(Enum):
  OPEN = 0
  BLOCKED_PRE_EXISTING = 1
  BLOCKED_PLAYER_PLACED = 2
  START = 3
  GOAL = 4

class CellType(Enum):
  OPEN = 0
  BLOCKED = 1
  START = 2
  GOAL = 3

class PatheryEnv(gym.Env):
  metadata = {"render_modes": ["ansi"], "render_fps": 4}

  def zeroGrid(self):
    # Initialize grid with OPEN cells (which have value 0)
    self.grid = np.zeros(self.gridSize, dtype=np.int32)

  def randomPos(self):
    x = self.np_random.integers(low=0, high=self.gridSize[0], dtype=np.int32)
    y = self.np_random.integers(low=0, high=self.gridSize[1], dtype=np.int32)
    return (x,y)

  def __init__(self, render_mode=None):
    # Initialize grid size
    self.gridSize = (6, 13)

    self.zeroGrid()

    # Observation space: Each cell type is a discrete value
    self.observation_space = spaces.MultiDiscrete(np.full((self.gridSize[0], self.gridSize[1]), len(CellType)))

    # Possible actions are which 2d position to place a block in
    self.action_space = spaces.MultiDiscrete((self.gridSize[0], self.gridSize[1]))

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

  def _get_obs(self):
    mapping = {
      InternalCellType.OPEN.value: CellType.OPEN.value,
      InternalCellType.BLOCKED_PRE_EXISTING.value: CellType.BLOCKED.value,
      InternalCellType.BLOCKED_PLAYER_PLACED.value: CellType.BLOCKED.value,
      InternalCellType.START.value: CellType.START.value,
      InternalCellType.GOAL.value: CellType.GOAL.value
    }

    def transform(cell):
      return mapping[cell]

    vectorized_transform = np.vectorize(transform)
    return vectorized_transform(self.grid)

  def _get_info(self):
    return {
      'Path length': self.lastPathLength
    }

  def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    self.zeroGrid()

    # Set the number of blocks that the user can place
    self.remainingBlocks = 7

    # Randomize start/goal
    # self.startPos = self.randomPos()
    # self.goalPos = self.startPos
    # while self.goalPos == self.startPos:
    #   self.goalPos = self.randomPos()

    # Fixed start/goal
    self.startPos = (1,0)
    self.goalPos = (1,12)

    # Fixed pre-placed blocks
    self.grid[0][0] = InternalCellType.BLOCKED_PRE_EXISTING.value
    self.grid[2][0] = InternalCellType.BLOCKED_PRE_EXISTING.value
    self.grid[3][0] = InternalCellType.BLOCKED_PRE_EXISTING.value
    self.grid[4][0] = InternalCellType.BLOCKED_PRE_EXISTING.value
    self.grid[5][0] = InternalCellType.BLOCKED_PRE_EXISTING.value
    self.grid[0][12] = InternalCellType.BLOCKED_PRE_EXISTING.value
    self.grid[2][12] = InternalCellType.BLOCKED_PRE_EXISTING.value
    self.grid[3][12] = InternalCellType.BLOCKED_PRE_EXISTING.value
    self.grid[4][12] = InternalCellType.BLOCKED_PRE_EXISTING.value
    self.grid[5][12] = InternalCellType.BLOCKED_PRE_EXISTING.value

    self.grid[1][7] = InternalCellType.BLOCKED_PRE_EXISTING.value
    self.grid[2][9] = InternalCellType.BLOCKED_PRE_EXISTING.value
    self.grid[5][8] = InternalCellType.BLOCKED_PRE_EXISTING.value

    # Place the start in top left
    self.grid[self.startPos[0]][self.startPos[1]] = InternalCellType.START.value

    # Place the end in bottom right
    self.grid[self.goalPos[0]][self.goalPos[1]] = InternalCellType.GOAL.value

    self.lastPathLength = self.calculateShortestPath()

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
          if self.grid[next_position[0]][next_position[1]] in [InternalCellType.OPEN.value, InternalCellType.START.value, InternalCellType.GOAL.value] and next_position not in visited:
            # Add the next position to the queue and mark it as visited
            queue.append((next_position, pathLength+1))
            visited.add(next_position)
    
    # There is no path to the goal
    return 0


  def step(self, action):
    if self.grid[action[0]][action[1]] == InternalCellType.OPEN.value:
      self.grid[action[0]][action[1]] = InternalCellType.BLOCKED_PLAYER_PLACED.value
      self.remainingBlocks -= 1
    else:
      # Invalid position; reward is -1, episode terminates
      return self._get_obs(), -1, True, False, self._get_info()
    
    pathLength = self.calculateShortestPath()

    if pathLength == 0:
      # Blocks path; reward is -1, episode terminates
      return self._get_obs(), -1, True, False, self._get_info()

    terminated = self.remainingBlocks == 0
    reward = pathLength - self.lastPathLength
    self.lastPathLength = pathLength

    observation = self._get_obs()
    info = self._get_info()

    return observation, reward, terminated, False, info

  def render(self):
    if self.render_mode == "ansi":
      return self._render_ansi()

  def _render_ansi(self):
    ansi_map = {
      InternalCellType.OPEN: '░',                  # Open cells
      InternalCellType.BLOCKED_PRE_EXISTING: '█',  # Blocked by pre-existing
      InternalCellType.BLOCKED_PLAYER_PLACED: '#', # Blocked by player
      InternalCellType.START: 'S',                 # Start
      InternalCellType.GOAL: 'G'                   # Goal
    }
    top_border = "+" + "-" * (self.gridSize[1] * 2 - 1) + "+"
    output = top_border + '\n'
    for row in self.grid:
      output += '|' + '|'.join(ansi_map[InternalCellType(val)] for val in row) + '|\n'
    output += top_border + '\n'
    output += f'Remaining blocks: {self.remainingBlocks}'
    return output

  def close(self):
    pass
