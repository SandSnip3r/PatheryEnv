from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from collections import deque

class InternalCellType(Enum):
  OPEN = 0
  ROCK = 1
  WALL = 2
  START = 3
  GOAL = 4

class CellType(Enum):
  OPEN = 0
  BLOCKED = 1
  START = 2
  GOAL = 3

class PatheryEnv(gym.Env):
  metadata = {"render_modes": ["ansi"], "render_fps": 4}

  def resetGrid(self):
    # Initialize grid with OPEN cells (which have value 0)
    self.grid = np.zeros(self.gridSize, dtype=np.int32)

  def randomPos(self):
    row = self.np_random.integers(low=0, high=self.gridSize[0], dtype=np.int32)
    col = self.np_random.integers(low=0, high=self.gridSize[1], dtype=np.int32)
    return (row, col)

  def __init__(self, render_mode=None, random_start=False, random_rocks=False, random_checkpoints=False):
    # Initialize grid size
    self.gridSize = (9, 17)
    self.wallsToPlace = 14
    self.random_start = random_start
    self.random_rocks = random_rocks
    self.random_checkpoints = random_checkpoints
    self.resetGrid()

    self.maxCheckpointCount = 2

    # Observation space: Each cell type is a discrete value
    self.observation_space = spaces.MultiDiscrete(np.full((self.gridSize[0], self.gridSize[1]), len(CellType) + self.maxCheckpointCount))

    # Possible actions are which 2d position to place a wall in
    self.action_space = spaces.MultiDiscrete((self.gridSize[0], self.gridSize[1]))

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

  def _get_obs(self):
    mapping = {
      InternalCellType.OPEN.value: CellType.OPEN.value,
      InternalCellType.ROCK.value: CellType.BLOCKED.value,
      InternalCellType.WALL.value: CellType.BLOCKED.value,
      InternalCellType.START.value: CellType.START.value,
      InternalCellType.GOAL.value: CellType.GOAL.value
    }

    def transform(cell):
      if cell >= len(InternalCellType):
        return cell - (len(InternalCellType) - len(CellType))
      return mapping[cell]

    vectorized_transform = np.vectorize(transform)
    return vectorized_transform(self.grid)

  def _get_info(self):
    return {
      'Path length': self.lastPathLength
    }

  def addCheckpoint(self, row, col, checkpointIndex):
    """Adds a checkpoint. Returns the next checkpoint index."""
    if checkpointIndex - len(InternalCellType) >= self.maxCheckpointCount:
      raise ValueError(f'Too many checkpoints. Max: {self.maxCheckpointCount}; trying to add #{checkpointIndex - len(InternalCellType)+1}')
    self.checkpoints.append((row, col, checkpointIndex))
    self.grid[row][col] = checkpointIndex
    return checkpointIndex + 1

  def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    self.resetGrid()

    # Set the number of walls that the user can place
    self.remainingWalls = self.wallsToPlace

    if self.random_start:
      # Choose a random start
      self.startPos = (self.np_random.integers(low=0, high=self.gridSize[0], dtype=np.int32),0)
    else:
      # Choose a fixed start
      self.startPos = (1,0)

    # Place all goals on the far right
    self.goalPositions = [(row, 16) for row in range(self.gridSize[0])]

    # Place the start
    self.grid[self.startPos[0]][self.startPos[1]] = InternalCellType.START.value

    # Place the goals
    for goalPos in self.goalPositions:
      self.grid[goalPos[0]][goalPos[1]] = InternalCellType.GOAL.value

    # Fixed pre-placed rocks (near start)
    for row in range(self.gridSize[0]):
      if row != self.startPos[0]:
        self.grid[row][0] = InternalCellType.ROCK.value

    self.checkpoints = []
    if self.random_checkpoints:
      self.generateCheckpoints(checkpointCount=self.maxCheckpointCount)
    else:
      # Place checkpoints
      next_index = self.addCheckpoint(5, 13, len(InternalCellType))
      next_index = self.addCheckpoint(1, 10, next_index)

    if self.random_rocks:
      self.placeRandomRocks(rocksToPlace=14)
    else:
      self.grid[2][1] = InternalCellType.ROCK.value
      self.grid[6][1] = InternalCellType.ROCK.value
      self.grid[3][3] = InternalCellType.ROCK.value
      self.grid[3][4] = InternalCellType.ROCK.value
      self.grid[1][5] = InternalCellType.ROCK.value
      self.grid[3][6] = InternalCellType.ROCK.value
      self.grid[8][6] = InternalCellType.ROCK.value
      self.grid[1][7] = InternalCellType.ROCK.value
      self.grid[3][9] = InternalCellType.ROCK.value
      self.grid[8][9] = InternalCellType.ROCK.value
      self.grid[1][11] = InternalCellType.ROCK.value
      self.grid[3][11] = InternalCellType.ROCK.value
      self.grid[4][11] = InternalCellType.ROCK.value
      self.grid[6][12] = InternalCellType.ROCK.value

    self.lastPathLength = self.calculateShortestPath()

    observation = self._get_obs()
    info = self._get_info()

    return observation, info

  def generateCheckpoints(self,checkpointCount):
    checkpointVal = len(InternalCellType)
    while checkpointCount>0:
      row, col = self.randomPos()

      # Check if the cell is open
      if self.grid[row][col] != InternalCellType.OPEN.value:
        continue
      
      # Place the checkpoint
      checkpointVal = self.addCheckpoint(row, col, checkpointVal)
      checkpointCount -= 1

    
  def placeRandomRocks(self, rocksToPlace:int):
    """Generates a random grid where it is possible to reach the end"""
    while rocksToPlace > 0:
      # Generate a random position
      randomRow, randomCol = self.randomPos()

      # Can only place rocks in open cells
      if self.grid[randomRow][randomCol] != InternalCellType.OPEN.value:
        continue
      
      # Place the rock and test if a path still exists
      self.grid[randomRow][randomCol] = InternalCellType.ROCK.value
      shortestPath = self.calculateShortestPath()
      if shortestPath != 0:
        # Success
        rocksToPlace -= 1
      else:
        # Failed to place here, reset the cell
        self.grid[randomRow][randomCol] = InternalCellType.OPEN.value

  def calculateShortestSubpath(self, subStartPos, goalType):
    print(f'Finding shortest path from {subStartPos} to goal {goalType}')
    # Directions for moving: right, left, down, up
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # Create a queue for BFS and add the starting point
    queue = deque([(subStartPos, 0)])
    
    # Set of visited nodes
    visited = set()
    visited.add(subStartPos)
    
    while queue:
      # Get the current position and the path length to it
      (current, pathLength) = queue.popleft()
      
      # If the current position is the goal, return the path
      if self.grid[current[0]][current[1]] == goalType:
        return pathLength
      
      # Explore all the possible directions
      for direction in directions:
        # Calculate the next position
        next_position = (current[0] + direction[0], current[1] + direction[1])
        
        # Check if the next position is within the grid bounds
        if (0 <= next_position[0] < self.gridSize[0]) and (0 <= next_position[1] < self.gridSize[1]):
          # Check if the next position is not an obstacle and not visited
          # print(f'Next position: {next_position}')
          if self.grid[next_position[0]][next_position[1]] not in [InternalCellType.ROCK.value, InternalCellType.WALL.value] and next_position not in visited:
            # print('  adding')
            # Add the next position to the queue and mark it as visited
            queue.append((next_position, pathLength+1))
            visited.add(next_position)
    
    # There is no path to the goal
    return 0

  def calculateShortestPath(self):
    if len(self.checkpoints) == 0:
      return self.calculateShortestSubpath(self.startPos, InternalCellType.GOAL.value)

    sum = self.calculateShortestSubpath(self.startPos, self.checkpoints[0][2])
    if sum == 0:
      # If any path is blocked, the entire path length is 0
      return 0
    for i in range(1, len(self.checkpoints)):
      calculatedPathLength = self.calculateShortestSubpath(self.checkpoints[i-1], self.checkpoints[i][2])
      if calculatedPathLength == 0:
        # If any path is blocked, the entire path length is 0
        return 0
      sum += calculatedPathLength
    calculatedPathLength = self.calculateShortestSubpath(self.checkpoints[-1], InternalCellType.GOAL.value)
    if calculatedPathLength == 0:
      # If any path is blocked, the entire path length is 0
      return 0
    sum += calculatedPathLength
    return sum

  def step(self, action):
    if self.grid[action[0]][action[1]] == InternalCellType.OPEN.value:
      self.grid[action[0]][action[1]] = InternalCellType.WALL.value
      self.remainingWalls -= 1
    else:
      # Invalid position; reward is -1, episode terminates
      return self._get_obs(), -1, True, False, self._get_info()
    
    pathLength = self.calculateShortestPath()

    if pathLength == 0:
      # Blocks path; reward is -1, episode terminates
      return self._get_obs(), -1, True, False, self._get_info()

    terminated = self.remainingWalls == 0
    reward = pathLength - self.lastPathLength
    if reward < 0:
      print(f'last path len: {self.lastPathLength}, this path length: {pathLength} obs:\n{self._get_obs()}')
      raise ValueError(f'Reward is negative: {reward}')
    self.lastPathLength = pathLength

    observation = self._get_obs()
    info = self._get_info()

    return observation, reward, terminated, False, info

  def render(self):
    if self.render_mode == "ansi":
      return self._render_ansi()

  def _render_ansi(self):
    ansi_map = {
      InternalCellType.OPEN: '░',  # Open cells
      InternalCellType.ROCK: '█',  # Blocked as a pre-existing part of the map
      InternalCellType.WALL: '#',  # Blocked by player
      InternalCellType.START: 'S', # Start
      InternalCellType.GOAL: 'G'   # Goal
    }
    def getChar(val):
      if val >= len(InternalCellType):
        # Return a character for checkpoints. First checkpoint is A, second is B, etc.
        return chr(ord('A') + val - len(InternalCellType))
      return ansi_map[InternalCellType(val)]

    top_border = "+" + "-" * (self.gridSize[1] * 2 - 1) + "+"
    output = top_border + '\n'
    for row in self.grid:
      output += '|' + '|'.join(getChar(val) for val in row) + '|\n'
    output += top_border + '\n'
    output += f'Remaining walls: {self.remainingWalls}'
    return output

  def close(self):
    pass
